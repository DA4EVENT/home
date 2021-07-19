import numpy as np
from scipy import io as sio

from evrepr.data.file_reader import FileReader


class MatFileReader(FileReader):
    ext = ".mat"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read_example(self, filename, start=0, count=-1):
        data = sio.loadmat(filename)
        if count < 0:
            count = data['x'].shape[0] - start  # all events

        x = data['x'][start:start+count].astype(np.float32)
        y = data['y'][start:start+count].astype(np.float32)
        ts = data['ts'][start:start+count].astype(np.float32)
        p = data['pol'][start:start+count].astype(np.float32)

        return x, y, ts, p
