import numpy as np

from evrepr.data.file_reader import FileReader


class RPGNpzReader(FileReader):
    ext = ".npz"

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def read_example(self, filename, start=0, count=-1):

        with np.load(filename) as data:
            x, y, p = data['xyp'].T
            t = data['t'].ravel()
        t = t * 1e6

        if count < 0:
            count = x.shape[0] - start  # all events

        x = x[start:start+count]
        y = y[start:start+count]
        t = t[start:start+count]
        p = p[start:start+count]

        return np.float32(x), np.float32(y), np.float32(t), np.float32(p)
