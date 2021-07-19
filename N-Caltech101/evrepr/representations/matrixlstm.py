import torch

import numpy as np

from .event_representation import EventRepresentation

from evrepr.thirdparty.matrixlstm.classification.layers.MatrixLSTM import MatrixLSTM
from evrepr.thirdparty.matrixlstm.classification.layers.SELayer import SELayer


class MatrixLSTMRepresentation(EventRepresentation):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.matrixlstm = MatrixLSTM(
            input_shape=tuple(cfg.frame_size),
            region_shape=tuple(cfg.region_shape),
            region_stride=tuple(cfg.region_stride),
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            bias=cfg.bias,
            lstm_type=cfg.lstm_type,
            add_coords_feature=cfg.add_coords_features,
            add_time_feature_mode=cfg.add_feature_mode,
            normalize_relative=cfg.normalize_relative,
            max_events_per_rf=cfg.max_events_per_rf,
            maintain_in_shape=cfg.maintain_in_shape,
            keep_most_recent=cfg.keep_most_recent,
            frame_intervals=cfg.frame_intervals,
            frame_intervals_mode=cfg.frame_intervals_mode
        )

        self.selayer = None
        if cfg.add_selayer:
            self.selayer = SELayer(self.matrixlstm.out_channels,
                                   reduction=1)

    @property
    def output_shape(self):
        return [self.matrixlstm.out_channels, *self.matrixlstm.output_shape]

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

    def forward(self, batched_inputs, batched_states=None):
        rois = self.get_active_regions(batched_inputs)
        events, lengths = self.preprocess_events(batched_inputs)

        if lengths.max() == 0:
            return events.new_zeros([events.shape[0], *self.output_shape])
        del batched_inputs

        coords = events[:, :, 0:2].type(torch.int64)
        ts = events[:, :, 2].float().unsqueeze(-1)
        embed = events[:, :, 3].float().unsqueeze(-1)

        images = self.matrixlstm(input=(embed, coords, ts, lengths))
        images = images.permute(0, 3, 1, 2)

        if self.selayer:
            images = self.selayer(images)

        return images, rois, None
