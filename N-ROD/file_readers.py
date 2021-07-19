from evrepr.data import (BinFileReader, ESIMFileReader, AedatFileReader,
                         RPGNpzReader as RPGNpzFileReader,
                         DatReader as DatFileReader)
from evrepr.data.transform import ClockwiseRotation
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import os


class Reader(ABC):

    def __init__(self, is_source, args):
        self.is_source = is_source
        self.args = args
        self._reader = None

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    @abstractmethod
    def __call__(self, path, **read_args):
        pass


class EventReader(Reader):
    is_image = False
    ext = []

    def __init__(self, is_source, args):
        super().__init__(is_source, args)

        prefix = 'source_' if is_source else 'target_'
        self.subsample_mode = getattr(self.args,
                                      prefix + "subsample_mode")
        self.subsample_value = getattr(self.args,
                                       prefix + "subsample_value")
        self.subsample_threshold = getattr(self.args,
                                           prefix + "subsample_threshold")

    def _subsample_events(self, events):
        old_nevents = events.shape[0]

        if self.subsample_mode is None:
            # If no mode is specified, we don't subsample
            return events

        if self.subsample_threshold is not None and \
                old_nevents <= self.subsample_threshold:
            # If a threshold is provided, we only subsample
            # if the threshold is exceeded
            return events

        new_nevents = int(self.subsample_value)
        if self.subsample_mode == "relative":
            assert 0 < self.subsample_value <= 1
            # If relative, value is the percentage to keep
            new_nevents = int(old_nevents * float(self.subsample_value))

        if new_nevents >= old_nevents:
            return events

        sample_index = np.linspace(0, old_nevents - 1, new_nevents,
                                   dtype=np.int64)
        return events[sample_index]

    def __call__(self, path, **read_args):
        x, y, t, p = self._reader.read_example(path, **read_args)
        events = np.stack([x, y, t, p], axis=-1).astype(np.float32)
        events = self._subsample_events(events)


        w = events[:, 0].max() + 1 if events.size > 0 else 0
        h = events[:, 1].max() + 1 if events.size > 0 else 0
       # w = events[:, 0].max() + 1
       # h = events[:, 1].max() + 1


        #assert t.size == 0 or (t.max() < 350e3 and t[t.shape[0]//2] > 1e3), \
        #    "The event timestamps seem not to be on the right temporal scale " + path + " "+str(t.max()) +" " + str(t.min())


        return {'events': events, 'path': path, 'w': w, 'h': h}


class RGBReader(Reader):
    is_image = True
    ext = ['.jpg', '.png']

    def __call__(self, path, **read_args):
        img = Image.open(path).convert("RGB")
        return img


class NumpyReader(Reader):
    is_image = True
    ext = '.npy'

    def __call__(self, path, **read_args):
        img = np.load(path)
        return img


class BinReader(EventReader):
    is_image = False
    ext = '.bin'

    def __init__(self, is_source, args):
        super().__init__(is_source, args)
        self._reader = BinFileReader()


class AedatReader(EventReader):
    is_image = False
    ext = '.aedat'

    def __init__(self, is_source, args):
        super().__init__(is_source, args)
        self._reader = AedatFileReader()
        self._transform = ClockwiseRotation(-90)

    def __call__(self, path, **read_args):
        data = super().__call__(path, **read_args)
        data['events'] = self._transform(data['events'])
        return data


class RPGNpzReader(EventReader):
    is_image = False
    ext = '.npz'

    def __init__(self, is_source, args):
        super().__init__(is_source, args)
        self._reader = RPGNpzFileReader()


class DatReader(EventReader):
    is_image = False
    ext = '_td.dat'

    def __init__(self, is_source, args):
        super().__init__(is_source, args)
        self._reader = DatFileReader()

    def __call__(self, path, **read_args):
        data = super().__call__(path, **read_args)
        events = data['events']
        # Remove ROI offset
        events[:, 0] = events[:, 0] - 112
        events[:, 1] = events[:, 1] - 52
        data['events'] = events
        return data


class ESIMReader(EventReader):
    is_image = False
    ext = '/images/'

    def __init__(self, is_source, args):
        super().__init__(is_source, args)
        self._reader = ESIMFileReader(
            contrast_threshold_pos=args.esim_threshold_range[0],
            contrast_threshold_neg=args.esim_threshold_range[0],
            refractory_period=args.esim_refractory_period,
            log_eps=args.esim_log_eps,
            use_log=args.esim_use_log)

    def __call__(self, path, **read_args):
        rand_th = np.random.uniform(*self.args.esim_threshold_range)
        #print("C threshold --> ", rand_th)
        self._reader.esim.setParameters(
            float(rand_th), float(rand_th),
            float(self.args.esim_refractory_period),
            float(self.args.esim_log_eps),
            self.args.esim_use_log)
        if self.args.esim_timestamps_path is not None:
            read_args['timestamps'] = os.path.join(
                path, self.args.esim_timestamps_path)

        return super().__call__(path, **read_args)


def get_reader(ext, is_source, args):

    # Search in all Reader's subclasses the one having
    # the class' 'ext' attribute matching the requested one
    for subcls in Reader.get_subclasses():
        if (isinstance(subcls.ext, str) and ext == subcls.ext) or \
           (isinstance(subcls.ext, list) and ext in subcls.ext):
            return subcls(is_source, args)
    raise ValueError("File extension '{}' is unknown".format(ext))
