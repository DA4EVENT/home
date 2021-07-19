import os
import sys
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset

from src.io.psee_loader import PSEELoader


class PseeDataset(Dataset):

    def __init__(self, root, transform=None):
        self.paths = glob(os.path.join(root, "**/*.dat"), recursive=True)
        self.names = set([os.path.basename(os.path.dirname(p))
                          for p in self.paths])
        self.name2id = dict(zip(sorted(self.names), range(len(self.names))))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        name = os.path.basename(os.path.dirname(path))

        reader = PSEELoader(path)
        events = reader.load_n_events(reader.event_count())
        label = self.name2id[name]

        if self.transform:
            events = self.transform(events)

        return events, label


class CenterTransform:

    def __init__(self, frame_shape):
        self.h, self.w = frame_shape

    def __call__(self, events):
        events['x'] += (self.w - events['x'].max()) // 2
        events['y'] += (self.h - events['y'].max()) // 2
        return events


def collate_fn(batch):

    events, lengths, labels = [], [], []
    features = ['x', 'y', 't', 'p']
    for ev, lbl in batch:

        events.append(np.stack([ev[f].astype(np.float32)
                                for f in features], axis=-1))
        lengths.append(ev['x'].shape[0])
        labels.append(lbl)

    max_length = max(lengths)
    events = [np.pad(ev, ((0, max_length-ln), (0, 0)), mode='constant')
              for ln, ev in zip(lengths, events)]
    events = torch.as_tensor(np.stack(events, axis=0))
    lengths = torch.as_tensor(lengths)
    labels = torch.as_tensor(labels)

    return events, lengths, labels
