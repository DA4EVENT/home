import torch
import torch.nn.functional as F

from evrepr import representations
from evrepr.utils.logger import setup_logger
logger = setup_logger(__name__)


def repeat_counts(x, counts):
    """
    Repeat each element x[i] a number of times contained in counts[i]
    :param torch.Tensor x: a tensor of shape [N] containing the values to repeat
    :param torch.Tensor counts: a tensor of shape [N] containing the repeat
        counts
    :return:
    """
    incr = x.new_zeros(counts.sum().item())
    set_idx = F.pad(torch.cumsum(counts[:-1], dim=0), [1, 0])
    set_val = x - F.pad(x[:-1], [1, 0])

    incr[set_idx] = set_val
    return torch.cumsum(incr, dim=0)


def multiple_shape(shape, stride):
    """
    Given a shape (tuple), it returns the closest larger bounding box whose
    sides are multiple of stride
    """
    return tuple((s + (stride - 1)) // stride * stride for s in shape)


def get_representation(cfg, *args, **kwargs):
    if hasattr(representations, cfg.name):
        logger.info("Creating {}".format(cfg.name))
        repr_cls = getattr(representations, cfg.name)
        return repr_cls(cfg.args, *args, **kwargs)
    else:
        raise ValueError("'{}' is not a valid event representation"
                         .format(cfg.name))
