import torch
from torch import nn

import numpy as np

from .event_representation import EventRepresentation
from hats_pytorch import HATS


class HATSRepresentation(EventRepresentation):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.bins = cfg.bins
        self.minibatch = cfg.minibatch or 1
        self.hats = HATS(input_shape=cfg.frame_size,
                         r=cfg.r,
                         k=cfg.k,
                         tau=cfg.tau,
                         delta_t=cfg.delta_t,
                         bins=cfg.bins,
                         fold=True)

    @property
    def output_shape(self):
        H = self.hats.grid_n_height * (self.hats.r * 2 + 1)
        W = self.hats.grid_n_width * (self.hats.r * 2 + 1)
        return (2 * self.bins, H, W)

    def _preprocess_events_dict(self, batched_inputs):
        events, events_lens = [], []

        for i, inputs in enumerate(batched_inputs):
            events.append(inputs['events'])
            events_lens.append(inputs['events'].shape[0])

        max_length = max(events_lens)
        events = [np.pad(ev, ((0, max_length - ln), (0, 0)),
                         mode='constant', constant_values=0) for
                  ln, ev in zip(events_lens, events)]
        events = torch.as_tensor(np.stack(events, axis=0), device=self.device)
        events_lens = torch.as_tensor(events_lens, device=self.device)

        return events, events_lens

    def _preprocess_events_torch(self, batched_inputs):
        return batched_inputs

    def get_active_regions(self, batched_inputs):
        rois = super().get_active_regions(batched_inputs)
        if self.input_is_torch:
            rois = (rois // self.hats.k) * (self.hats.r * 2 + 1)
        else:
            rois = [tuple(int((p // self.hats.k) * (self.hats.r * 2 + 1))
                          for p in points) for points in rois]
        return rois

    def forward(self, batched_inputs, batched_states=None):
        rois = self.get_active_regions(batched_inputs)
        events, lengths = self.preprocess_events(batched_inputs)

        if lengths.max() == 0:
            return events.new_zeros([events.shape[0], *self.output_shape])
        del batched_inputs

        # FIXME: Temporary fix for OOM on large batches. We split the
        #   batch in minibatches, process them sequentially and then
        #   concatenate the results
        minibatch_events = torch.split(events, self.minibatch, dim=0)
        minibatch_lengths = torch.split(lengths, self.minibatch, dim=0)
        minibatch_images = []
        for events, lengths in zip(minibatch_events, minibatch_lengths):
            events = events[:, :lengths.max()].contiguous()
            minibatch_images.append(self.hats(events, lengths))
        images = torch.cat(minibatch_images, dim=0)

        return images, rois, None
