import os
import numpy as np
from numba import njit

from evrepr.data.file_reader import FileReader


@njit
def _overflow_idx(array, ovf_th):
    prev = 0
    for idx, val in np.ndenumerate(array):
        if idx[0] > 0 and int(prev) - int(val) > ovf_th:
            return idx[0]
        prev = val
    return -1


class DatReader(FileReader):
    ext = ".dat"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read_example(self, filename, start=0, count=-1):

        #                ts                nothing  p     y         x
        # |______________________________||________|_|_________|__________|
        #              32bit                      1bit  14bit     14bit

        with open(filename, 'rb') as f:
            # If comment section is not present,
            # version 1.0 is assumed by the standard
            headers = {}
            has_comments = False
            prev = 0
            line = f.readline().decode("utf-8", "ignore")
            # Reads the comments and extracts the aer format version
            while line.startswith('%'):
                line = line[:-1] if line[-1] == '\n' else line
                words = line.split(' ')
                if len(words) > 2:
                    has_comments = True
                    if words[1] == 'Date':
                        if len(words) > 3:
                            headers[words[1]] = words[2] + ' ' + words[3]
                    else:
                        headers.update({words[1]: words[2:]})
                prev = f.tell()
                line = f.readline().decode("utf-8", "ignore")
            # Repositions the pointer at the beginning of data section
            f.seek(prev)

            if has_comments:
                ev_type = int.from_bytes(f.read(1), byteorder='little')
                ev_size = int.from_bytes(f.read(1), byteorder='little')
            else:
                ev_type = 0
                ev_size = 8

            # Moves the file pointer at the 'start' position
            # based on the event size
            f.seek(start * ev_size, os.SEEK_CUR)
            # Reads count events (-1, default value, means all)
            raw_data = np.fromfile(f, dtype=np.uint32,
                                   count=max(-1, count * ev_size//4)).\
                newbyteorder('<')

        all_ts = raw_data[0::1+(ev_size-4)//4].astype(np.float)
        all_addr = raw_data[1::1+(ev_size-4)//4]

        version = int(headers.get('Version', ["0"])[0])
        xmask = 0x00003FFF
        ymask = 0x0FFFC000
        polmask = 0x10000000
        xshift = 0
        yshift = 14
        polshift = 28

        all_addr = np.abs(all_addr)
        all_x = ((all_addr & xmask) >> xshift).astype(np.float)
        all_y = ((all_addr & ymask) >> yshift).astype(np.float)
        all_p = ((all_addr & polmask) >> polshift).astype(np.float)
        length = len(all_x)

        # check overflow
        ovf_th = 500000  # 500 ms
        idx_overflow = _overflow_idx(all_ts, ovf_th=ovf_th)
        if idx_overflow > 0:
            all_ts[idx_overflow:] += 2 ** 32

        return all_x, all_y, all_ts, all_p

