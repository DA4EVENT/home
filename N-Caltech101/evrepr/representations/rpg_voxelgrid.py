import torch
from torch import nn

from .event_representation import EventRepresentation


def events_to_voxel_grid_mod(events_torch, num_bins, width, height):
    """
    A slightly modified version of thirdparty.e2vid.utils.inference_utils,
    where the input is already been placed on a torch device
    Code from: https://github.com/uzh-rpg/rpg_e2vid
    """

    device = events_torch.device
    assert (events_torch.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    with torch.no_grad():
        voxel_grid = torch.zeros(num_bins, height, width,
                                 dtype=torch.float32,
                                 device=device).flatten()

        # Normalize the event timestamps so that they lie
        # between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * \
                             (events_torch[:, 0] - first_stamp) / deltaT
        ts = events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices]
                                    * width
                                    + tis_long[valid_indices]
                                    * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices]
                                    * width
                                    + (tis_long[valid_indices] + 1)
                                    * width * height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid


class RPGVoxelGridSurface(nn.Module):

    def __init__(self, frame_shape, bins):
        super().__init__()
        self.h, self.w = frame_shape
        self.bins = bins

    def forward(self, events_list, device):

        voxel_grids = [events_to_voxel_grid_mod(
            events_torch=events,
            num_bins=self.bins,
            height=self.h,
            width=self.w)
            for events in events_list]

        # voxel_grids.shape = [B, NBins, H, W]
        voxel_grids = torch.stack(voxel_grids, dim=0)
        return voxel_grids


class RPGVoxelGridRepresentation(EventRepresentation):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.bins = cfg.bins
        self.frame_shape = cfg.frame_size
        self.surface = RPGVoxelGridSurface(cfg.frame_size, cfg.bins)

    @property
    def output_shape(self):
        return (self.bins, *self.frame_shape)

    def _preprocess_events_dict(self, batched_inputs):
        events_list, empty_seqs = [], []
        for i, inputs in enumerate(batched_inputs):
            if inputs['events'].shape[0] > 0:
                ev = inputs['events']
                ev = ev[:, [2, 0, 1, 3]]  # (ts, x, y, p)
                ev = torch.as_tensor(ev, device=self.device)
                events_list.append(ev)
            else:
                empty_seqs.append(i)

        return events_list, empty_seqs

    def _preprocess_events_torch(self, batched_inputs):
        events_list, empty_seqs = [], []
        for i, (ev, n) in enumerate(zip(*batched_inputs)):
            if n.item() > 0:
                ev = ev[:n, [2, 0, 1, 3]]  # (ts, x, y, p)
                events_list.append(ev)
            else:
                empty_seqs.append(i)

        return events_list, empty_seqs

    def forward(self, batched_inputs, batched_states=None):
        rois = self.get_active_regions(batched_inputs)
        events_list, empty_seqs = self.preprocess_events(batched_inputs)

        if len(events_list) == 0:
            return torch.zeros([len(batched_inputs), *self.output_shape],
                               device=self.device, dtype=torch.float32)
        del batched_inputs

        repr = self.surface(events_list, self.device)
        repr = self.fill_empty(repr, empty_seqs)
        return repr, rois, None
