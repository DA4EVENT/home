from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from evrepr.utils.logger import setup_logger


class EventRepresentation(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for event representations.
    """

    def __init__(self, cfg, input_is_torch=False):

        super().__init__()
        self.cfg = cfg
        self.logger = setup_logger(name=self.__class__.__name__)
        self.register_buffer("device_token", torch.empty(size=[]))
        # If input_is_torch = True we expect a pytorch tensor of padded
        #   events as input, otherwise a list of dictionaries as returned
        #   by the leader class
        self.input_is_torch = (cfg.input_is_torch
                               if hasattr(cfg, "input_is_torch") and
                                  cfg.input_is_torch is not None
                               else input_is_torch)

    @property
    def device(self):
        return self.device_token.device

    @abstractmethod
    def forward(self, batched_inputs, batched_states=None):
        """
        Subclasses must override this method, but adhere to the same return type

        Returns:
            Tuple[torch.Tensor, Any]: a tuple (images, shapes, state) containing
            the event representations, region of the output frame corresponding
            to active pixels and possibly the internal state
        """
        pass

    @property
    def output_shape(self):
        """
        Returns:
            Tuple
        """
        pass

    def preprocess_events(self, batched_inputs):
        """
        An utility method to preprocess events and make them comply with the
        representation's input encoding
        :param batched_inputs: if self.input_is_torch is True, this is a tuple:
              - torch.Tensor events: a [B, Tmax, 4] tensor containing padded
                  events
              - torch.Tensor lengths: a [B] tensor containing the length of each
                  sample before any padding is applied
            otherwise, if self.input_is_torch is False, this is a list of
            dictionaries, one for each sample. Each dict contains an 'events'
            key containing the unpadded events as a [T, 4] np.ndarray
        :return:
        """
        if self.input_is_torch:
            assert isinstance(batched_inputs, (list, tuple)) \
                   and len(batched_inputs) == 2
            assert isinstance(batched_inputs[0], torch.Tensor) and \
                   isinstance(batched_inputs[1], torch.Tensor)
            return self._preprocess_events_torch(batched_inputs)
        else:
            assert isinstance(batched_inputs, (list, tuple))
            assert isinstance(batched_inputs[0], dict) and \
                   'events' in batched_inputs[0]
            return self._preprocess_events_dict(batched_inputs)

    @abstractmethod
    def _preprocess_events_torch(self, batched_inputs):
        pass

    @abstractmethod
    def _preprocess_events_dict(self, batched_inputs):
        pass

    def get_active_regions(self, batched_inputs):
        """
        Compute the region of the output representation that received at least
        one event. Batched inputs may have different resolutions, this method
        can be used to determine which part of each sample is padding, and which
        is not (i.e., the active part)

        Returns:
            A list of tuples (x1, y1, x2, y2) defining the active region,
            one for each sample
        """

        if self.input_is_torch:
            # Events have been padded to Tmax and stacked on the first dim
            # batched_inputs = (padded_events, lengths)
            # padded_events.shape = [B, Tmax, 4], lengths.shape = [B]
            return torch.tensor(
                [(0, 0,
                  int(ev[:n, 0].max() if n.item() > 0 else 0),
                  int(ev[:n, 1].max() if n.item() > 0 else 0))
                 for ev, n in zip(*batched_inputs)],
                device=batched_inputs[0].device, dtype=torch.int64)
        else:
            # No collate has already been applied, the batch is a list of dicts,
            # one per sample, with an 'event' key containing the Ti x 4 events
            # batched_inputs = [{'events': np.ndarray}, ...]
            return [(0, 0,
                     int(i['events'][:, 0].max() if i['events'].size > 0 else 0),
                     int(i['events'][:, 1].max() if i['events'].size > 0 else 0))
                    for i in batched_inputs]

    def fill_empty(self, x, empty_ids, fill_x=None, fill_value=0):
        """
        Event representations are computed only for samples having at least
        one event. This utility can be used to insert empty event
        representations for those samples that have not been computed. Their
        sample id must be provided as 'empty_ids', while 'x' contains
        actual representations from non-empty samples. If a 'fill_x' [B, ...]
        tensor is provided, it will be used to fill the missing representations
        instead of creating an empty one. This procedure can also be used to
        fill representation's states
        """

        num_empty = len(empty_ids)
        if num_empty == 0:
            return x

        empty_ids = sorted(empty_ids)
        if fill_x is not None:
            fill_values = fill_x[empty_ids]
        else:
            fill_values = x.new_full([num_empty, *x.shape[1:]], fill_value)

        for i, seq_id in enumerate(empty_ids):
            x = torch.cat([x[:seq_id], fill_values[[i]], x[seq_id:]])

        return x
