import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

import numpy as np
from itertools import product

from .utils import repeat_counts
from .event_representation import EventRepresentation


class EventVolumeSurface(nn.Module):
    """
    Implement the Events Volume surface as presented in:
    Zhu, Alex Zihao, et al. "Unsupervised event-based learning of optical flow,
    depth, and egomotion. "Proceedings of the IEEE Conference on Computer Vision
    and Pattern Recognition. 2019.
    """

    def __init__(self, frame_size, bins):
        super().__init__()
        self.h, self.w = frame_size
        self.bins = bins

    @staticmethod
    def k_b(a):
        return torch.relu(1 - torch.abs(a))

    def forward(self, events, lengths):
        # Events is a tensor [N, 5] containing all events
        # in the batch as (x, y, t, p, batch_id)
        x, y, t, p, b = events.t()
        B = int(b[-1].item()) + 1

        # Normalize the events
        # The events t0, t1 and tT of each sample in the batch
        last_idx = torch.cumsum(lengths, dim=0) - 1
        first_idx = F.pad(last_idx[:-1] + 1, [1, 0])
        t0 = repeat_counts(t[first_idx], lengths)
        tN = repeat_counts(t[last_idx], lengths)
        t_star = (self.bins - 1) * torch.clamp((t - t0) / (tN - t0), 0, 1)

        # Implement the equation:
        # V(x,y,t) = \sum_i{ pi * kb(x - xi) * kb(y - yi) * kb(t - ti*)}
        # Instead of performing a cycle for every (x, y, t, i) we just consider
        # the 4 integer coordinates around each event coordinate and the two
        # bins around the event timestamp, which will give a nonzero k_b()
        vox = 0  # dummy value, will be a [B, C, H, W] tensor
        round_fns = (torch.floor, torch.ceil)
        xybin_round_fns = product(round_fns, round_fns, round_fns)
        for x_round, y_round, bin_round in xybin_round_fns:
            bin_ref = bin_round(t_star)
            x_ref = x_round(x)
            y_ref = y_round(y)

            # Avoid summing the same contribution multiple times if the
            # pixel or time coordinate is already an integer. In that
            # case both floor and ceil provide the same ref. If it is an
            # integer, we only add it if the case #_round is torch.floor
            # We also remove any out of frame or bin coordinate due to ceil
            valid_ref = (((x_ref != x).bool() | (x_round is torch.floor)) &
                         ((y_ref != y).bool() | (y_round is torch.floor)) &
                         ((bin_ref != t_star).bool() | (bin_round is torch.floor)) &
                         (x_ref < self.w).bool() & (y_ref < self.h).bool() &
                         (bin_ref < self.bins).bool())
            x_ref = x_ref[valid_ref]
            y_ref = y_ref[valid_ref]
            bin_ref = bin_ref[valid_ref]

            # Compute the contribution
            val = p[valid_ref] * \
                  self.k_b(x_ref - x[valid_ref]) * \
                  self.k_b(y_ref - y[valid_ref]) * \
                  self.k_b(bin_ref - t_star[valid_ref])

            # Add the contribution in position (bin_ref, x_ref, y_ref) of
            # the final voxel
            idx = x_ref \
                  + y_ref * self.w \
                  + bin_ref * self.h * self.w \
                  + b[valid_ref] * self.bins * self.h * self.w

            vox += scatter(src=val, index=idx.long(),
                           dim_size=B * self.bins * self.h * self.w,
                           reduce='sum')

        vox = vox.reshape(B, self.bins, self.h, self.w)

        return vox


class EventVolumeRepresentation(EventRepresentation):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.frame_size = cfg.frame_size
        self.bins = cfg.bins

        self.surface = EventVolumeSurface(frame_size=self.frame_size,
                                          bins=self.bins)

    @property
    def output_shape(self):
        return (self.bins, *self.frame_size)

    def _preprocess_events_dict(self, batched_inputs):
        events, events_lens, empty_seqs = [], [], []
        for i, inputs in enumerate(batched_inputs):
            if inputs['events'].shape[0] > 0:
                ev = inputs['events']
                ev[ev[:, 3] < 1, 3] = -1

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
                ev[ev[:, 3] < 1, 3] = -1

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
            events = torch.empty([0, 5], device=pad_events.device,
                                 dtype=torch.float32)
            events_lens = torch.empty([0], device=pad_events.device,
                                      dtype=torch.int64)

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

