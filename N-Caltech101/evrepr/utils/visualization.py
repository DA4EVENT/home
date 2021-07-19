import numpy as np
from collections import Mapping


def channels_to_rgb(ev_repr, clip_outliers):
    H, W, C = ev_repr.shape
    assert C in [1, 2, 3]

    if C == 1:
        # Grayscale
        ev_repr = np.repeat(ev_repr, 3, axis=-1)
    elif C == 2:
        # Red and blue
        ev_repr = np.stack(
            [ev_repr[..., 0],
             np.zeros_like(ev_repr[..., 0]),
             ev_repr[..., 1]], axis=-1)

    im_max = ev_repr.max() if not clip_outliers else ev_repr.std() * 5
    img = np.clip((ev_repr - ev_repr.min()) / (im_max - ev_repr.min()), 0, 1)
    img = (255 * img).astype(np.uint8)

    return img


def postprocess_representation(ev_repr, group_size, clip_outliers):
    assert ev_repr.ndim == 3
    assert group_size in [1, 2, 3]

    # ev_repr.shape [H, W, C]
    ev_repr = ev_repr.transpose(1, 2, 0)
    H, W, C = ev_repr.shape

    # Group channels and convert each group to rgb
    ev_reprs = np.split(
        ev_repr, list(range(group_size, C, group_size)), axis=-1)
    ev_reprs = [channels_to_rgb(r, clip_outliers) for r in ev_reprs]

    image = np.concatenate(ev_reprs, axis=1)
    return image


def pformat_dict(d, indent=0):
    fstr = ""
    for key, value in d.items():
        fstr += '\n' + '  ' * indent + str(key) + ":"
        if isinstance(value, Mapping):
            fstr += pformat_dict(value, indent+1)
        else:
            fstr += ' ' + str(value)
    return fstr
