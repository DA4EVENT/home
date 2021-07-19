import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from .utils import repeat_counts
from .event_representation import EventRepresentation


class TBRSurface(nn.Module):
    """
    Implement the Temporal Binary Representation as presented in:
    Innocenti, Simone Undri, et al. "Temporal Binary Representation for
    Event-Based Action Recognition." arXiv preprint arXiv:2010.08946 (2020).
    """

    def __init__(self, frame_size, bins):
        super().__init__()
        self.h, self.w = frame_size
        self.bins = bins

        pow = torch.arange(self.bins).reshape(self.bins, 1, 1)
        pow = pow.repeat(1, self.h, self.w)
        pow = (2 ** pow).unsqueeze(0)  # [1, bins, h, w]
        self.register_buffer("pow", pow)

    @property
    def output_shape(self):
        H = self.hats.grid_n_height * (self.hats.r * 2 + 1)
        W = self.hats.grid_n_width * (self.hats.r * 2 + 1)
        return (2 * self.hats.bins, H, W)

    def forward(self, events, lengths):
        # Events is a tensor [N, 5] containing all events
        # in the batch as (x, y, t, p, batch_id)
        x, y, t, p, b = events.t()
        B = int(b[-1].item()) + 1
        device = events.device

        # Normalize the events
        # The events t0 and tT of each sample in the batch
        last_idx = torch.cumsum(lengths, dim=0) - 1
        first_idx = F.pad(last_idx[:-1] + 1, [1, 0])
        t0 = repeat_counts(t[first_idx], lengths)
        tN = repeat_counts(t[last_idx], lengths)
        t_norm = (t - t0) / (tN - t0 + 1e-5)
        # Compute the bin id associated to each event
        t_bin = torch.clamp((t_norm * self.bins).int(), 0, self.bins - 1)

        # Compute the binary masks
        idx = x \
              + y * self.w \
              + t_bin * self.h * self.w \
              + b * self.bins * self.h * self.w
        val = torch.ones_like(x, device=device, dtype=torch.float32)

        vox_bin = torch.zeros([B, self.bins, self.h, self.w],
                              dtype=torch.float32, device=device)
        vox_bin.put_(idx.long(), val, accumulate=False)

        # Convert binary to decimal
        vox_dec = (vox_bin * self.pow).sum(1, keepdim=True) / (2 ** self.bins)

        return vox_dec


class TBRRepresentation(EventRepresentation):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.frame_size = cfg.frame_size
        self.bins = cfg.bins

        self.surface = TBRSurface(frame_size=self.frame_size,
                                  bins=self.bins)

    @property
    def output_shape(self):
        return (1, *self.frame_size)

    def _preprocess_events_dict(self, batched_inputs):
        events, events_lens, empty_seqs = [], [], []
        for i, inputs in enumerate(batched_inputs):
            if inputs['events'].shape[0] > 0:
                ev = inputs['events']
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

