import numpy as np
from torchvision.transforms import Compose


def get_transforms(cfg):
    transforms = []
    if cfg is not None:
        for trf_cfg in cfg:
            if trf_cfg.name in globals():
                trf_cls = globals()[trf_cfg.name]
                trf = trf_cls(**dict(trf_cfg.args or {}))
                transforms.append(trf)
            else:
                raise ValueError("Transofrm {} does not exist!"
                                 "".format(trf_cfg.name))
    return Compose(transforms)


class ClockwiseRotation(object):

    def __init__(self, deg):
        self.deg = deg

    def __repr__(self):
        return "{0.__class__.__name__}(def={0.deg})".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """

        x, y, ts, p = np.split(events, 4, axis=-1)
        # Compute the center of the events cloud
        xc = (x.max() - x.min()) / 2
        yc = (y.max() - y.min()) / 2

        # Apply rotation
        angle = np.radians(self.deg)
        x_rot = ((x - xc) * np.cos(angle)) - ((y - yc) * np.sin(angle)) + xc
        y_rot = ((x - xc) * np.sin(angle)) + ((y - yc) * np.cos(angle)) + yc

        # Translate events so that the top-left most event is in (0,0)
        x_left = np.min(x_rot)
        y_top = np.min(y_rot)
        x_rot -= x_left
        y_rot -= y_top

        x_rot = np.around(x_rot).astype(np.int32)
        y_rot = np.around(y_rot).astype(np.int32)

        return np.column_stack([x_rot, y_rot, ts, p])


class HorizontalFlip(object):

    def __repr__(self):
        return "{0.__class__.__name__}()".format(self)

    def __call__(self, events):
        """
        :param np.ndarray events: [num_events, 4] array containing
            (x, y, ts, p) values
        :return: np.ndarray [num_events, 4]
        """

        x, y, ts, p = np.split(events, 4, axis=-1)
        x = x.max() + x.min() - x
        return np.column_stack([x, y, ts, p])
