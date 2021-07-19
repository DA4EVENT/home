import os
from glob import glob

import esim_py
from evrepr.data.file_reader import FileReader


class ESIMFileReader(FileReader):
    ext = ".png"

    def __init__(self,
                 contrast_threshold_pos,
                 contrast_threshold_neg,
                 refractory_period,
                 log_eps,
                 use_log,
                 timestamps=None,
                 **kwargs):

        self.timestamps = timestamps
        self.esim = esim_py.EventSimulator(
            contrast_threshold_pos,
            contrast_threshold_neg,
            refractory_period,
            log_eps,
            use_log,
        )

        super().__init__(**kwargs)

    @classmethod
    def glob(cls, root_path):
        img_paths = glob(
            os.path.join(root_path, "**/*" + cls.ext),
            recursive=True
        )

        dirname_paths = set(os.path.dirname(p) for p in img_paths)
        dirname_paths = list(sorted(dirname_paths))

        return dirname_paths

    def read_example(self, dirname, start=0, count=-1, timestamps=None):

        timestamps = timestamps or self.timestamps
        if not timestamps:
            same_txts = glob(os.path.join(dirname, "*.txt"))
            parent_txts = glob(os.path.join(dirname, "../*.txt"))
            if len(same_txts) == 1:
                timestamps = same_txts[0]
            elif len(parent_txts) == 1:
                timestamps = parent_txts[0]
            else:
                raise RuntimeError(
                    "Could not find a timestamps file for sample {}"
                    "".format(dirname))

        events = self.esim.generateFromFolder(dirname, timestamps)

        if count < 0:
            count = events.shape[0] - start  # all events
        events = events[start:start+count]

        x, y, t, p = events.T
        # We divide the timestamp by 1e3 as timestamps.txt
        # are usually in ns
        # FIXME: this should be handled in a better way
        t = t / 1e3
        p[p == -1] = 0

        return x, y, t, p
