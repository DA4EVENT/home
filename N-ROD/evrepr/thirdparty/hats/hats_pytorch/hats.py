import torch
from torch import nn
from torch.nn import functional as F
from hats_cuda import group_by_cell, group_by_bin, local_surface


class HATS(nn.Module):

    def __init__(self, input_shape=(180, 240), r=3, k=10,
                 tau=1e6, delta_t=100000, bins=1, fold=False):
        """
        Implements the HATS event representation as described in "HATS:
        Histograms of Averaged Time Surfaces for Robust Event-based Object
        Classification", Sironi et al.

        :param tuple input_shape: the (height, width) size of the frame
        :param int r: the size of the 2r+1 x 2r+1 spatial neighborhood
        :param int k: the size of the cells
        :param float tau: the decay factor
        :param float delta_t: the delay defining the temporal neighborhood
        :param float bins: the number of bins to compute. The input
            sequence is split into 'bins' intervals, a representation is
            extracted from each of them, and representations are finally
            stacked on the channel dimension
        :param bool fold: if True, histograms are organized in a grid of
            shape [B, 2*bins, (2R+1) * NhCells, (2R+1) * NwCells] where each
            histogram is placed its original position in the frame. Otherwise,
            a [B, NCells, 2*bins, 2R+1, 2R+1] tensor is returned
        """
        super().__init__()

        self.fold = fold
        self.delta_t = delta_t
        self.bins = bins
        self.tau = tau
        self.r = r
        self.k = k

        # Make sure the input shape is divisible by k
        padded_shape = tuple((s + (k - 1)) // k * k for s in input_shape)
        self.input_shape = padded_shape
        height, width = padded_shape

        self.grid_n_width = (width // k)
        self.grid_n_height = (height // k)
        self.n_cells = self.grid_n_width * self.grid_n_height
        self.cell_size = 2*self.r + 1

        self.output_shape = (2, self.grid_n_height * self.cell_size,
                             self.grid_n_width * self.cell_size)

        self.register_buffer('coord2cellid', self.get_coord2cellid_matrix())

    def get_coord2cellid_matrix(self):
        """
        Compute a matrix [H, W] containing in each pixel location the id
        of the cell containing that pixel.
        E.g., given an 4x4 input space and k=2, this function returns:
            [[0 0 1 1],
             [0 0 1 1],
             [2 2 3 3],
             [2 2 3 3]]
        """

        flat_matrix = torch.arange(self.n_cells) \
            .repeat(self.k ** 2) \
            .view(1, -1, self.n_cells).float()
        pixels2rf_matrix = F.fold(flat_matrix, self.input_shape,
                                  kernel_size=(self.k, self.k), stride=self.k)
        return pixels2rf_matrix.type(torch.int64).view(self.input_shape)

    def split_bins(self, events, lengths):
        """
        Given a sequence of events, it splits each sequence into self.bins
        temporal intervals.

        :param torch.Tensor events: a [B, TPad, C] tensor of events
        :param torch.Tensor lengths: a [B] tensor of original sequence lengths
        :return: a tuple (split_events, split_lengths):
            - split_events: a [B*bins, TPad, C] tensor of events
            - split_lengths: a [B*bins] tensor of original sequence lengths
        """

        if self.bins <= 1:
            return events, lengths

        B, TPad, C = events.shape
        device = events.device

        t = events[:, :, 2].float()
        # Compute the time of the first and last events in each sample
        last_idx = (lengths - 1).long()
        batch_idx = torch.arange(B, device=device)

        first_ts = t[:, [0]]  # [B, 1]
        last_ts = t[batch_idx, last_idx].unsqueeze(-1)  # [B, 1]

        # Convert time into a bin id
        t_norm = (t - first_ts) / (last_ts - first_ts + 1e-6)
        t_bin = torch.clamp((t_norm * self.bins).int(), 0, self.bins - 1)

        # Compute the index for the sum
        batch_idx = torch.arange(B, device=device, dtype=torch.int64)
        batch_idx = batch_idx.unsqueeze(-1).repeat(1, TPad)
        idx = (batch_idx * self.bins + t_bin).reshape(-1)

        # We sum 0 if it is a padded event, 1 otherwise
        values = torch.arange(lengths.max(), device=device, dtype=torch.int32)
        values = (values.repeat(B, 1) < lengths.unsqueeze(1)).reshape(-1)

        # Count the number of events in each bin
        bins_count = torch.zeros([B, self.bins],
                                 dtype=torch.int64, device=device)
        bins_count.put_(idx.long(), values.long(), accumulate=True)

        # We now group events into bins
        split_events = group_by_bin(events, bins_count)
        split_lengths = bins_count.reshape(-1)

        return split_events, split_lengths

    def group_bins(self, hists):
        """
        Group histograms extracted from different intervals together by
        placing them on the channel dimension

        :param hists: if self.fold = True, histograms are organized in a grid of
            shape [B*bins, 2, (2R+1) * NhCells, (2R+1) * NwCells] where each
            histogram is placed its original position in the frame. Otherwise,
            a [B*bins, NCells, 2, 2R+1, 2R+1] tensor is returned
        :return: if self.fold = True, histograms are organized in a grid of
            shape [B, bins*2, (2R+1) * NhCells, (2R+1) * NwCells] where each
            histogram is placed its original position in the frame. Otherwise,
            a [B, NCells, bins*2, 2R+1, 2R+1] tensor is returned
        """

        if self.bins <= 1:
            return hists

        if not self.fold:
            # hists.shape = [B*bins, NCells, 2, 2R+1, 2R+1]
            Bbins, NCells, _, S, _ = hists.shape
            hists = hists.reshape(-1, self.bins, NCells, 2, S, S)
            hists = hists.permute(0, 2, 1, 3, 4, 5)
            hists = hists.reshape(-1, NCells, self.bins * 2, S, S)
        else:
            # hists.shape = [B*bins, 2, (2R+1) * NhCells, (2R+1) * NwCells]
            Bbins, _, Sh, Sw = hists.shape
            hists = hists.reshape(-1, self.bins * 2, Sh, Sw)
        return hists

    def cells_memory(self, events, lengths):
        """
        Compute the memory of each cell, that is, the set of events contained in
        in each cell. This function maintains th original event order when
        splitting events into cells. Note: cells of different samples are
        grouped in the same dimension and cells with no events are discarded.
        In the case all cells in all samples contain at least one event,
        B*NCells memory cells are extracted.

        :param torch.Tensor events: a [B, TPad, C] tensor of events
        :param torch.Tensor lengths: a [B] tensor of original sequence lengths
        :return: A tuple (cell_b, cell_len, cell_h, cell_w, cell_memory):
            - cell_b: a [B*NCells] tensor containing, for each extracted memory,
                the original sample id from which it has been extracted
            - cell_len: a [B*NCells] tensor containing the number of events
                in each memory cell
            - cell_h: a [B*NCells] tensor containing for each cell, its y
                position in the grid of cells
            - cell_w: a [B*NCells] tensor containing for each cell, its x
                position in the grid of cells
            - cell_memory: [TCellPad, B*NCells, C] tensor, when TCellPad is
                the temporal dimension, eventually padded with zero values at
                the end, B*NCells is the total number of active cells in the
                batch (i.e., receiving at least one event) and C is the event's
                channel dimension. Note, only the cells containing at least one
                event are extracted.
        """

        # Make sure lengths are int64
        lengths = lengths.long()

        # Retrieve the cell id in which events are contained
        ids = self.coord2cellid[events[..., 1].type(torch.int64),
                                events[..., 0].type(torch.int64)]

        # Group events by cell id, maintaining the original event order
        # groups.shape = [TCellPad, B*NCells, C]
        return group_by_cell(events, ids, lengths,
                             self.grid_n_height, self.grid_n_width)

    def cells_histograms(self, cell_memory, cell_len):
        """
        Compute the normalized histograms for each cell
        :param torch.Tensor cell_memory: a [TCellPad, B*NCells, C] tensor
            containing for each  B*NCells cell, a list of TCellPad events with
            C channels, optionally padded at the end with zero values
        :param torch.Tensor cell_len: a [B*NCells] tensor containing the actual
            length of each memory cell before padding
        :return: a [BNCells, 2, 2R+1, 2R+1] tensor representing the two surfaces
            one for each polarity of all the cells
        """

        # Compute the surface for each event
        # [TCellPad, BNCells, 2, 2R+1, 2R+1]
        cell_local_surfaces = local_surface(cell_memory, cell_len,
                                            self.delta_t, self.r, self.tau)
        # Sum the event's surfaces together
        # [TCellPad, BNCells, 2, 2R+1, 2R+1] -> [BNCells, 2, 2R+1, 2R+1]
        cell_local_surfaces = cell_local_surfaces.sum(0)

        # Avoid div by zero, reshape [BNCells, 1, 1, 1]
        cell_len = torch.clamp_min(cell_len, 1).reshape(-1, 1, 1, 1)
        # Normalize by the event count
        cell_histograms = cell_local_surfaces / cell_len

        return cell_histograms

    def group_histograms(self, cell_hists, cell_b, cell_h, cell_w, batch_size):
        """
        The histogram extraction process is performed independently on each
        memory cell, regardless of the sample id or position of the original
        cell. This function groups histograms back together based on the sample
        they come from, and the position of the cell in the frame.
        :param torch.Tensor cell_hists: the [BNCells, 2, 2R+1, 2R+1] histograms
            to be grouped
        :param torch.Tensor cell_b: a [B*NCells] tensor containing, for each
            histogram, the original sample id from which it has been extracted
        :param torch.Tensor cell_h: a [B*NCells] tensor containing for each
            histogram, its y position in the grid of cells
        :param torch.Tensor cell_w: a [B*NCells] tensor containing for each
            histogram, its x position in the grid of cells
        :param int batch_size: the original batch size
        :return: if self.fold = True, histograms are organized in a grid of
            shape [B, 2, (2R+1) * NhCells, (2R+1) * NwCells] where each
            histogram is placed its original position in the frame. Otherwise,
            a [B, NCells, 2, 2R+1, 2R+1] tensor is returned
        """

        # Unfold the histograms back together
        hist_unfold = cell_hists.new_zeros(
            [batch_size, self.grid_n_height, self.grid_n_width, 2,
             self.cell_size, self.cell_size])
        hist_unfold[cell_b, cell_h, cell_w] = cell_hists

        if not self.fold:
            hist = hist_unfold.reshape(batch_size, -1, 2, self.cell_size,
                                       self.cell_size)
        else:
            hist_unfold = hist_unfold.permute(0, 3, 4, 5, 1, 2) \
                .reshape(batch_size, 2 * self.cell_size ** 2, self.n_cells)

            hist = F.fold(hist_unfold, self.output_shape[-2:],
                          kernel_size=self.cell_size,
                          stride=self.cell_size)
        return hist

    def forward(self, events, lengths):
        """
        :param torch.Tensor events: the input events organized as a tensor of
            shape [B, TPad, C], where B is the batch dimension, TPad is the
            temporal dimension (padded at the end with zero values if sample b
            is shorter), and C is the channel dimension containing (x, y, t, p)
            features
        :param torch.Tensor lengths: a tensor of shape [B] specifying for each
            sample in "events", its original length before padding
        :return: if self.fold = True, histograms are organized in a grid of
            shape [B, 2, (2R+1) * NhCells, (2R+1) * NwCells] where each
            histogram is placed its original position in the frame. Otherwise,
            a [B, NCells, 2, 2R+1, 2R+1] tensor is returned
        """

        # Split events into bins, placing bins in the batch dimension,
        # they will then be stacked back together at the end
        events, lengths = self.split_bins(events, lengths)
        B, TPad, C = events.shape

        # cell_memory.shape = [TCellPad, B*NCells, C]
        cell_b, cell_len, cell_h, cell_w, cell_memory = \
            self.cells_memory(events, lengths)

        # Compute the local surfaces [BNCells, 2, 2R+1, 2R+1]
        cell_hists = self.cells_histograms(cell_memory, cell_len)

        # Group back the surfaces based on sample id and cell id.
        # Optionally reorder them in a grid if self.fold = True
        hists = self.group_histograms(cell_hists, cell_b, cell_h, cell_w, B)
        # Group back the bins based on the sample they were in
        hists = self.group_bins(hists)

        return hists
