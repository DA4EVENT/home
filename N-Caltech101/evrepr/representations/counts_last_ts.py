import torch
from torch import nn
from torch_scatter import scatter

import numpy as np
from abc import abstractmethod, ABC

from .utils import repeat_counts
from .event_representation import EventRepresentation


class HandCraftedSurface(nn.Module, ABC):

    def __init__(self, frame_size, bins):
        super().__init__()
        self.h, self.w = frame_size
        self.bins = bins

    def get_idx(self, events, lengths):
        # Events is a tensor [N, 5] containing all events
        # in the batch as (x, y, t, p, batch_id)
        x, y, t, p, b = events.t()
        B = int(b[-1].item()) + 1

        # We assume that if 0 <= ts <= 1 timestamps have
        # already been normalized
        if t.min() < 0 or t.max() > 1:
            # The max ts of each sample  (shape [B])
            b_maxs = scatter(src=t, index=b, dim_size=B, reduce='max')
            # Assign to each event the corresponding max based on the sample id
            maxs = repeat_counts(b_maxs, lengths)
            t = t / maxs

        bin = torch.clamp((t * self.bins).int(), 0, self.bins - 1)
        idx = x \
              + y * self.w \
              + p * self.h * self.w \
              + bin * 2 * self.h * self.w \
              + b * self.bins * 2 * self.h * self.w

        return idx.long()

    @abstractmethod
    def forward(self, events, lengths):
        pass


class LastTsSurface(HandCraftedSurface):

    def forward(self, events, lengths, idx=None):
        B = int(events[-1, -1].item()) + 1

        idx = idx if idx is not None else self.get_idx(events, lengths)
        vox_ts = scatter(src=events[:, 2], index=idx,
                         dim_size=B * self.bins * 2 * self.h * self.w,
                         reduce='max')
        vox_ts = vox_ts.reshape(B, self.bins * 2, self.h, self.w)
        return vox_ts


class CountsSurface(HandCraftedSurface):

    def forward(self, events, lengths, idx=None):
        B = int(events[-1, -1].item()) + 1

        idx = idx if idx is not None else self.get_idx(events, lengths)
        vox_cnts = scatter(src=events.new_ones([events.shape[0]]), index=idx,
                           dim_size=B * self.bins * 2 * self.h * self.w,
                           reduce='sum')
        vox_cnts = vox_cnts.reshape(B, self.bins, 2, self.h * self.w)
        vox_maxs = vox_cnts.max(dim=-1, keepdim=True).values
        vox_cnts = vox_cnts / vox_maxs
        vox_cnts = vox_cnts.reshape(B, self.bins * 2, self.h, self.w)

        return vox_cnts


class CountsLastTsSurface(HandCraftedSurface):

    def __init__(self, frame_size, bins, features=('last_ts', 'counts')):
        super().__init__(frame_size, bins)
        self.features = features
        self.ts_surf = LastTsSurface(frame_size, bins)
        self.cnts_surf = CountsSurface(frame_size, bins)

    def forward(self, events, lengths, idx=None):
        B = int(events[-1, -1].item()) + 1

        voxs = []
        idx = idx if idx is not None else self.get_idx(events, lengths)

        if 'counts' in self.features:
            voxs.append(self.cnts_surf(events, lengths, idx))
        if 'last_ts' in self.features:
            voxs.append(self.ts_surf(events, lengths, idx))

        vox = torch.cat([vox.reshape(B, self.bins, 2, self.h, self.w)
                         for vox in voxs], dim=2)
        vox = vox.reshape(B, -1, self.h, self.w)

        return vox


class CountsLastTsRepresentation(EventRepresentation):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.frame_size = cfg.frame_size
        self.bins = cfg.bins
        self.features = cfg.features

        self.surface = CountsLastTsSurface(frame_size=self.frame_size,
                                           bins=self.bins,
                                           features=self.features)

    @property
    def output_shape(self):
        return (self.bins * 2 * len(self.features), *self.frame_size)

    def _preprocess_events_dict(self, batched_inputs):
        events, events_lens, empty_seqs = [], [], []
        for i, inputs in enumerate(batched_inputs):
            if inputs['events'].shape[0] > 0:
                ev = inputs['events']
                ev[:, 2] = ev[:, 2] / ev[:, 2].max()

                events.append(np.concatenate([
                    ev, np.full([ev.shape[0], 1], len(events))], axis=-1))
                events_lens.append(ev.shape[0])
            else:
                empty_seqs.append(i)

        if len(events) != 0:
            events = torch.as_tensor(np.concatenate(events, axis=0),
                                     device=self.device, dtype=torch.float32)
            events_lens = torch.as_tensor(events_lens, device=self.device)
        else:
            events = torch.empty(
                [0, 5], device=self.device, dtype=torch.float32)
            events_lens = torch.empty(
                [0], device=self.device, dtype=torch.int64)

        return events, events_lens, empty_seqs

    def _preprocess_events_torch(self, batched_inputs):
        pad_events, events_lens = batched_inputs
        events, empty_seqs = [], []
        for i, (ev, n) in enumerate(zip(*batched_inputs)):
            if n.item() > 0:
                ev = ev[:n]
                ev[:, 2] = ev[:, 2] / ev[:, 2].max()

                events.append(torch.cat([
                    ev, torch.full([ev.shape[0], 1], len(events),
                                   dtype=ev.dtype, device=ev.device)],
                    dim=-1))
            else:
                empty_seqs.append(i)

        if len(events) != 0:
            events = torch.cat(events, dim=0).float()
            events_lens = events_lens[events_lens > 0]
        else:
            events = torch.empty(
                [0, 5], device=pad_events.device, dtype=torch.float32)
            events_lens = torch.empty(
                [0], device=pad_events.device, dtype=torch.int64)

        return events, events_lens, empty_seqs

    def forward(self, batched_inputs, batched_states=None):
        rois = self.get_active_regions(batched_inputs)
        events, lengths, empty_seqs = self.preprocess_events(batched_inputs)

        if events.shape[0] == 0:
            return events.new_zeros([len(batched_inputs), *self.output_shape])
        del batched_inputs

        repr = self.surface(events, lengths)
        repr = self.fill_empty(repr, empty_seqs)
        return repr, rois, None
