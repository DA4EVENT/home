import os
import numpy as np

from evrepr.data.file_reader import FileReader


class AedatFileReader(FileReader):
    ext = ".aedat"

    def __init__(self, device="DVS128", **kwargs):
        self.device = device
        super().__init__(**kwargs)

    def _get_camera_format(self):

        if self.device == "DVS128":
            x_mask = 0xFE
            x_shift = 1
            y_mask = 0x7F00
            y_shift = 8
            p_mask = 0x1
            p_shift = 0
        else:
            raise ValueError("Unsupported camera: {}".format(self.device))

        return x_mask, x_shift, y_mask, y_shift, p_mask, p_shift

    def _read_aedat20_events(self, f, count=-1):

        raw_data = np.fromfile(f, dtype=np.int32, count=max(-1, count * 2)).\
            newbyteorder('>')
        f.close()

        all_data = raw_data[0::2]
        all_ts = raw_data[1::2]

        # Events' representation depends of the camera format
        x_mask, x_shift, y_mask, y_shift, p_mask, p_shift = \
            self._get_camera_format()

        all_x = ((all_data & x_mask) >> x_shift).astype(np.int32)
        all_y = ((all_data & y_mask) >> y_shift).astype(np.int32)
        all_p = ((all_data & p_mask) >> p_shift).astype(np.int32)
        all_ts = all_ts.astype(np.int32)
        length = len(all_x)

        return length, all_x, all_y, all_ts, all_p

    def _read_aedat31_events(self, f, count=-1):

        # WARNING: This function assumes that all the events are of type
        # POLARITY_EVENT and so each packet has a fixed size and structure.
        # If your dataset may contain other type of events you must write a
        # function to properly handle different packets' sizes and formats.
        # See: https://inilabs.com/support/software/fileformat/#h.w7vjqzw55d5b

        raw_data = np.fromfile(f, dtype=np.int32, count=max(-1, count * 2))
        f.close()

        all_x, all_y, all_ts, all_p = [], [], [], []

        while raw_data.size > 0:

            # Reads the header
            block_header, raw_data = raw_data[:7], raw_data[7:]
            eventType = block_header[0] >> 16
            eventSize, eventTSOffset, eventTSOverflow, \
            eventCapacity, eventNumber, eventValid = block_header[1:]
            size_events = eventNumber * eventSize // 4
            events, raw_data = raw_data[:size_events], raw_data[size_events:]

            if eventValid and eventType == 1:
                data = events[0::2]
                ts = events[1::2]

                x = ((data >> 17) & 0x1FFF).astype(np.int32)
                y = ((data >> 2) & 0x1FFF).astype(np.int32)
                p = ((data >> 1) & 0x1).astype(np.int32)
                valid = (data & 0x1).astype(np.bool)
                ts = ((eventTSOverflow.astype(np.int64) << 31) | ts).\
                    astype(np.int64)

                # The validity bit can be used to invalidate events.
                # We filter out the invalid ones
                if not np.all(valid):
                    x = x[valid]
                    y = y[valid]
                    ts = ts[valid]
                    p = p[valid]

                all_x.append(x)
                all_y.append(y)
                all_ts.append(ts)
                all_p.append(p)

        all_x = np.concatenate(all_x, axis=-1)
        all_y = np.concatenate(all_y, axis=-1)
        all_ts = np.concatenate(all_ts, axis=-1)
        all_p = np.concatenate(all_p, axis=-1)
        length = len(all_x)

        return length, all_x, all_y, all_ts, all_p

    def read_example(self, filename, start=0, count=-1):
        f = open(filename, 'rb')

        # If comment section is not present, version 1.0 is assumed by
        # the standard
        version = "1.0"
        prev = 0
        line = f.readline().decode("utf-8", "ignore")
        # Reads the comments and extracts the aer format version
        while line.startswith('#'):
            if line[0:9] == '#!AER-DAT':
                version = line[9:12]
            prev = f.tell()
            line = f.readline().decode("utf-8", "ignore")
        # Repositions the pointer at the beginning of data section
        f.seek(prev)

        # Moves the file pointer at the 'start' position based on the event size
        f.seek(start * 4 * 2, os.SEEK_CUR)  # ev_size = 2 * 32bit (2*4*8bit)

        # Reads count events (-1, default values, means all),
        # i.e. (count * 5) bytes
        if version == "2.0":
            length, all_x, all_y, all_ts, all_p = \
                self._read_aedat20_events(f, count=count)
        elif version == "3.1":
            length, all_x, all_y, all_ts, all_p = \
                self._read_aedat31_events(f, count=count)
        else:
            raise NotImplementedError("Reader for version {} has not "
                                      "yet been implemented.".format(version))

        return all_x, all_y, all_ts, all_p
