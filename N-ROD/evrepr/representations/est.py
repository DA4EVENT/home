import torch
from torch import nn

import numpy as np

from .event_representation import EventRepresentation
from evrepr.thirdparty.est.utils.models import QuantizationLayer


class ESTRepresentation(EventRepresentation):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        act_args = cfg.activation_args
        self.est = QuantizationLayer(
            dim=[cfg.bins, *cfg.frame_size],
            mlp_layers=cfg.mlp_layers,
            activation=getattr(nn, cfg.activation)(**act_args)
        )

    @property
    def output_shape(self):
        C, H, W = self.est.dim
        return (2*C, H, W)

    def _preprocess_events_dict(self, batched_inputs):
        events, empty_seqs = [], []
        for i, inputs in enumerate(batched_inputs):
            if inputs['events'].shape[0] > 0:
                ev = inputs['events']
                ev[ev[:, 3] < 1, 3] = -1

                events.append(np.concatenate([
                    ev, np.full([ev.shape[0], 1], len(events))],
                    axis=-1))
            else:
                empty_seqs.append(i)

        if len(events) != 0:
            events = torch.as_tensor(np.concatenate(events, axis=0),
                                     device=self.device, dtype=torch.float32)
        else:
            events = torch.empty([0, 5], device=self.device,
                                 dtype=torch.float32)
        return events, empty_seqs

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
        else:
            events = torch.empty([0, 5], device=pad_events.device,
                                 dtype=torch.float32)
        return events, empty_seqs

    def forward(self, batched_inputs, batched_states=None):
        rois = self.get_active_regions(batched_inputs)
        events, empty_seqs = self.preprocess_events(batched_inputs)

        if events.shape[0] == 0:
            return events.new_zeros([len(batched_inputs), *self.output_shape])
        del batched_inputs

        repr = self.est(events)
        repr = self.fill_empty(repr, empty_seqs)
        return repr, rois, None
