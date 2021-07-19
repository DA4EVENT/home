import numpy as np

from evrepr.data.file_reader import FileReader


class BinFileReader(FileReader):
    ext = ".bin"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read_example(self, filename, start=0, count=-1):
        f = open(filename, 'rb')
        # Moves the file pointer at the 'start' position based on the event size
        f.seek(start * 5)  # ev_size = 40bit (5 * 8bit)
        # Reads count events (-1, default values, means all),
        # i.e. (count * 5) bytes
        raw_data = np.fromfile(f, dtype=np.uint8, count=max(-1, count * 5))
        f.close()
        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7
        all_ts = ((raw_data[2::5] & 127) << 16) | \
                 (raw_data[3::5] << 8) | \
                 (raw_data[4::5])

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        x = np.array(all_x[td_indices], dtype=np.int32)
        y = np.array(all_y[td_indices], dtype=np.int32)
        ts = np.array(all_ts[td_indices], dtype=np.int32)
        p = np.array(all_p[td_indices], dtype=np.int32)

        return x, y, ts, p
