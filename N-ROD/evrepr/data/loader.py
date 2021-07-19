import os
import torch
import numpy as np
import multiprocessing

from torch.utils.data import Dataset, DataLoader

from evrepr import data
from evrepr.data.transform import get_transforms
from evrepr.utils.logger import setup_logger
logger = setup_logger(__name__)


class EventDataset(Dataset):

    def __init__(self, cfg):
        self.root_path = cfg.path
        self.transform = get_transforms(cfg.transforms)
        self.reader = get_reader(cfg.reader)
        self.file_paths = self.reader.glob(self.root_path)

        self.names = set([os.path.basename(os.path.dirname(p))
                          for p in self.file_paths])
        self.name2id = dict(zip(sorted(self.names), range(len(self.names))))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        name = os.path.basename(os.path.dirname(path))
        x, y, t, p = self.reader.read_example(path)

        events = np.stack([x, y, t, p], axis=-1).astype(np.float32)
        if self.transform is not None:
            events = self.transform(events)

        lbl = self.name2id[name]
        num_events = x.shape[0]
        w = int(x.max() + 1) if num_events > 0 else 0
        h = int(y.max() + 1) if num_events > 0 else 0

        return {'events': events,
                'class_id': lbl,
                'class_name': name,
                'path': path,
                'w': w,
                'h': h}


def get_reader(cfg):
    if hasattr(data, cfg.name):
        logger.info("Using {}".format(cfg.name))
        reader_cls = getattr(data, cfg.name)
        reader_args = cfg.args or {}
        return reader_cls(**reader_args)
    else:
        raise ValueError("'{}' is not a valid event reader"
                         .format(cfg.name))


def collate_fn(batch):
    # Do not collate, we let the event representation
    # aggregate samples as different representations need
    # different input formats
    return batch


def collate_padevents_fn(batch):
    events, events_lens = [], []

    for i, inputs in enumerate(batch):
        events.append(inputs['events'])
        events_lens.append(inputs['events'].shape[0])

    max_length = max(events_lens)
    events = [np.pad(ev, ((0, max_length - ln), (0, 0)),
                     mode='constant', constant_values=0) for
              ln, ev in zip(events_lens, events)]
    events = torch.as_tensor(np.stack(events, axis=0))
    events_lens = torch.as_tensor(events_lens)

    return events, events_lens


def map_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple):
        return tuple(map_to_device(d, device) for d in data)
    elif isinstance(data, list):
        return list(map_to_device(d, device) for d in data)
    elif isinstance(data, dict):
        return {k: map_to_device(v, device) for k, v in data.items()}
    else:
        return data


def get_loader(cfg):

    dataset = EventDataset(cfg)
    logger.info("{} files detected".format(len(dataset)))

    loader_args = {
        'batch_size': 1,
        'collate_fn': collate_padevents_fn if cfg.to_torch else collate_fn,
        'num_workers': multiprocessing.cpu_count()
    }
    loader_args.update(dict(cfg.loader.args))
    loader = DataLoader(dataset, **loader_args)

    return loader
