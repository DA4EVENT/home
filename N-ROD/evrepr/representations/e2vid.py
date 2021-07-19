import torch

from .event_representation import EventRepresentation
from .utils import multiple_shape

from evrepr.utils.io import open_url
from evrepr.thirdparty.e2vid.model import model
from evrepr.thirdparty.e2vid.utils.inference_utils import (
    events_to_voxel_grid_pytorch
)
from .rpg_voxelgrid import events_to_voxel_grid_mod


class E2VidRepresentation(EventRepresentation):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.frame_shape = cfg.frame_size
        self.e2vid_frame_shape = multiple_shape(cfg.frame_size, 8)
        if cfg.weights is not None:
            self.e2vid = self._load_e2vid(cfg.weights)
        else:
            ev2vid_cls = getattr(model, cfg.arch)
            ev2vid_cfg = {
                'num_bins': cfg.bins,
                'skip_type': cfg.skip_type,
                'num_encoders': cfg.num_encoders,
                'base_num_channels': cfg.base_num_channels,
                'num_residual_blocks': cfg.num_residual_blocks,
                'norm': cfg.norm,
                'use_upsample_conv': cfg.use_upsample_conv
            }
            self.e2vid = ev2vid_cls(ev2vid_cfg)

    @property
    def output_shape(self):
        return (1, *self.frame_shape)

    def _load_e2vid(self, path):
        self.logger.info('Loading model {}...'.format(path))
        with open_url(path, "rb") as fp:
            raw_model = torch.load(fp)

        arch = getattr(model, raw_model['arch'])
        try:
            config = raw_model['model']
        except:
            config = raw_model['config']['model']

        e2vid = arch(config)
        e2vid.load_state_dict(raw_model['state_dict'])
        return e2vid

    def _preprocess_events_dict(self, batched_inputs):
        events_tensor, empty_seqs = [], []
        for i, inputs in enumerate(batched_inputs):
            if inputs['events'].shape[0] > 0:
                ev = inputs['events']
                ev = ev[:, [2, 0, 1, 3]]  # (ts, x, y, p)

                events_tensor.append(events_to_voxel_grid_pytorch(
                    events=ev,
                    num_bins=self.e2vid.num_bins,
                    height=self.e2vid_frame_shape[0],
                    width=self.e2vid_frame_shape[1],
                    device=self.device)
                )
            else:
                empty_seqs.append(i)

        if len(events_tensor) != 0:
            events_tensor = torch.stack(events_tensor,  dim=0)
        else:
            events_tensor = torch.empty(
                [0, self.e2vid.num_bins, *self.frame_shape],
                device=self.device, dtype=torch.float32)
        return events_tensor, empty_seqs

    def _preprocess_events_torch(self, batched_inputs):
        pad_events, _ = batched_inputs
        events_tensor, empty_seqs = [], []
        for i, (ev, n) in enumerate(zip(*batched_inputs)):
            if n.item() > 0:
                ev = ev[:n]
                ev = ev[:, [2, 0, 1, 3]]  # (ts, x, y, p)

                events_tensor.append(events_to_voxel_grid_mod(
                    events_torch=ev,
                    num_bins=self.e2vid.num_bins,
                    height=self.e2vid_frame_shape[0],
                    width=self.e2vid_frame_shape[1])
                )
            else:
                empty_seqs.append(i)

        if len(events_tensor) != 0:
            events_tensor = torch.stack(events_tensor,  dim=0)
        else:
            events_tensor = torch.empty(
                [0, self.e2vid.num_bins, *self.frame_shape],
                device=pad_events.device, dtype=torch.float32)
        return events_tensor, empty_seqs

    def fill_empty_states(self, x, empty_ids, fill_x=None, fill_value=0):
        num_empty = len(empty_ids)
        if num_empty == 0:
            return x

        assert isinstance(x, list)

        empty_ids = sorted(empty_ids)
        if fill_x is not None:
            fill_values = fill_x[empty_ids]
        else:
            val = (torch.full_like(s, fill_value) for s in x[0])
            fill_values = [val] * num_empty

        for i, seq_id in enumerate(empty_ids):
            x.insert(seq_id, fill_values[i])

        return x

    def forward(self, batched_inputs, batched_states=None):
        rois = self.get_active_regions(batched_inputs)
        events, empty_seqs = self.preprocess_events(batched_inputs)

        if events.shape[0] == 0:
            return events.new_zeros([len(batched_inputs), *self.output_shape])
        del batched_inputs

        repr, state = self.e2vid(events, batched_states)
        # Remove additional padding added in order to make the shape
        # a multiple of e2vid minimum stride
        repr = repr[..., :self.frame_shape[0], :self.frame_shape[1]]
        repr = self.fill_empty(repr, empty_seqs)
        state = self.fill_empty_states(state, empty_seqs, batched_states)
        return repr, rois, state
