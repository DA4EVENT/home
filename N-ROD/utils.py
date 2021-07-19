import re
import os
import platform
import functools
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch._six import int_classes, string_classes, container_abcs

def l2norm_loss_self_driven(x):
    radius = x.norm(p=2, dim=1).detach()
    norm = radius.mean()
    assert radius.requires_grad == False
    radius = radius + 1.0
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return l, norm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def entropy_loss(logits):
    p_softmax = F.softmax(logits, dim=1)
    mask = p_softmax.ge(0.000001)  # greater or equal to
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))


def set_channel(args, modality):
    n_channel = 0
    modality = modality.split("-")[1] if "-" in modality else modality
    print(modality)

    if (modality == "rgb") or (modality == "depth"):
        n_channel = 3
    elif modality == "event":
        n_channel = -1
    elif modality == "eventvolume":
        n_channel = 9
    elif modality == "voxelgrid_3chans":
        n_channel = 3
    elif modality == "voxelgrid_6chans":
        n_channel = 6
    elif modality == "voxelgrid_9chans":
        n_channel = 9
    elif modality == "hats":
        n_channel = 2
    elif args.evrepr == "EST":
        n_channel = args.est_bins * 2
    elif args.evrepr == "HATS":
        n_channel = 2 * args.hats_bins
    elif args.evrepr == "RPGVoxelGrid":
        n_channel = args.rpgvoxelgrid_bins



    return n_channel


class OptimizerManager:
    def __init__(self, optims, num_batch_num, num_accumulation):
        self.optims = optims  # if isinstance(optims, Iterable) else [optims]
        self.num_bs = num_batch_num
        self.num_accumulation = num_accumulation

    def __enter__(self):
        pass

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        if ( self.num_bs % self.num_accumulation) == 0:
            for op in self.optims:
                op.step()
                op.zero_grad()
        self.optims = None
        self.num_bs = None

        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


class EvaluationManager:
    def __init__(self, nets):
        self.nets = nets

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch._C.set_grad_enabled(False)
        for net in self.nets:
            net.eval()

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        for net in self.nets:
            net.train()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad


class IteratorWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def __iter__(self):
        self.iterator = iter(self.loader)

    def get_next(self):
        try:
            items = self.iterator.next()
        except:
            self.__iter__()
            items = self.iterator.next()
        return items


def default_param(args):

    print("{} -> {}".format(args.source, args.target))
    make_paths(dataset=args.dataset,source=args.source, target=args.target, args=args)

    if "-" in args.modality:
        print("Multi-Modality Mode")
        args.multimodal = True
    else:
        print("Single-Modality Mode")
        args.multimodal = False


def make_paths(dataset = "Cifar10", source = 'Real', target='Real',args=None):
    node = platform.node()
    print("{} -> {}".format(args.source, args.target))

    data_root_source, data_root_target = None, None
    train_file_source, test_file_source = None, None
    train_file_target, test_file_target = None, None

    if node == "tiche":
        if dataset in ["caltech101", "ncaltech101"]:
            if source in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_source = "/home/chiarap/event-data/CALTECH101/"
                args.train_file_source = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/train.txt"
                args.val_file_source = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/val.txt"
                args.test_file_source = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/test.txt"

            if target in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_target = "/home/chiarap/event-data/CALTECH101/"
                args.train_file_target = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/train.txt"
                args.val_file_target = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/val.txt"
                args.test_file_target = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/test.txt"

        elif dataset in ["cifar10", "cifar10dvs"]:
            if source in ["Real", "Syn","Sim"]:
                args.data_root_source = "/data/event-data/CIFAR10"
                args.train_file_source = "/data/event-data/CIFAR10/train_split.txt"
                args.val_file_source = "/data/event-data/CIFAR10/test_split.txt"
                args.test_file_source = "/data/event-data/CIFAR10/test_split.txt"

            if target in ["Real", "Syn"]:
                args.data_root_target = "/data/event-data/CIFAR10"
                args.train_file_target = "/data/event-data/CIFAR10/train_split.txt"
                args.val_file_target = "/data/event-data/CIFAR10/test_split.txt"
                args.test_file_target = "/data/event-data/CIFAR10/test_split.txt"

        elif dataset in ["ROD"]:
            args.class_num = 51
            if source in ["Real", "Syn"]:
                args.data_root_source = "/home/chiarap/event-data/ROD_evrepr/"
                args.train_file_source = "/home/mirco/SPLIT_ROD/synARID_50k-split_sync_train1.txt"
                args.val_file_source = "/home/mirco/SPLIT_ROD/synARID_50k-split_sync_test1.txt"
                args.test_file_source = "/home/mirco/SPLIT_ROD/synARID_50k-split_sync_test1.txt"

            if target in ["Real", "Syn"]:
                args.data_root_target = "/home/chiarap/event-data/ROD_evrepr/"
                args.train_file_target = "/home/mirco/SPLIT_ROD/wrgbd_40k-split_sync.txt"

            args.val_file_target = args.val_file_source #non abbiamo un val per questo setting
            args.test_file_target = "/home/mirco/SPLIT_ROD/wrgbd_40k-split_sync.txt"




    elif node == "demetra":
        if dataset in ["caltech101", "ncaltech101"]:
            if source in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_source = "/home/chiarap/event-data/CALTECH101/"
                args.train_file_source = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/train.txt"
                args.val_file_source = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/val.txt"
                args.test_file_source = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/test.txt"

            if target in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_target = "/home/chiarap/event-data/CALTECH101/"
                args.train_file_target = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/train.txt"
                args.val_file_target = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/val.txt"
                args.test_file_target = "/home/chiarap/event-data/CALTECH101/CALTECH101_split/test.txt"

        elif dataset in ["cifar10", "cifar10dvs"]:
            if source in ["Real", "Syn","Sim"]:
                args.data_root_source = "/data/event-data/CIFAR10"
                args.train_file_source = "/data/event-data/CIFAR10/train_split.txt"
                args.val_file_source = "/data/event-data/CIFAR10/test_split.txt"
                args.test_file_source = "/data/event-data/CIFAR10/test_split.txt"

            if target in ["Real", "Syn"]:
                args.data_root_target = "/data/event-data/CIFAR10"
                args.train_file_target = "/data/event-data/CIFAR10/train_split.txt"
                args.val_file_target = "/data/event-data/CIFAR10/test_split.txt"
                args.test_file_target = "/data/event-data/CIFAR10/test_split.txt"

        elif dataset in ["ROD"]:
            args.class_num = 51
            if source in ["Real", "Syn"]:
                args.data_root_source = "/home/mirco/ROD_evrepr/"
                args.train_file_source = "/home/mirco/SPLIT_ROD/synARID_50k-split_sync_train1.txt"
                args.val_file_source = "/home/mirco/SPLIT_ROD/synARID_50k-split_sync_test1.txt"
                args.test_file_source = "/home/mirco/SPLIT_ROD/synARID_50k-split_sync_test1.txt"

            if target in ["Real", "Syn"]:
                args.data_root_target = "/home/mirco/ROD_evrepr/"
                args.train_file_target = "/home/mirco/SPLIT_ROD/wrgbd_40k-split_sync.txt"

            args.val_file_target = args.val_file_source #non abbiamo un val per questo setting
            args.test_file_target = "/home/mirco/SPLIT_ROD/wrgbd_40k-split_sync.txt"



    elif os.environ["HOME"].split("/")[-1] == "mplanamente": #Franklin
        print("Franklin")
        if dataset in ["caltech101", "ncaltech101"]:
            if source in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_source = "/work/mplanamente/event-data/CALTECH101"
                args.train_file_source = "/work/mplanamente/event-data/CALTECH101_split/train.txt"
                args.val_file_source = "/work/mplanamente/event-data/CALTECH101_split/val.txt"
                args.test_file_source = "/work/mplanamente/event-data/CALTECH101_split/test.txt"

            if target in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_target = "/work/mplanamente/event-data/CALTECH101"
                args.train_file_target = "/work/mplanamente/event-data/CALTECH101_split/train.txt"
                args.val_file_target = "/work/mplanamente/event-data/CALTECH101_split/val.txt"
                args.test_file_target = "/work/mplanamente/event-data/CALTECH101_split/test.txt"

        elif dataset in ["cifar10", "cifar10dvs"]:
            if source in ["Real", "Syn"]:
                args.data_root_source = "/home/chiarap/event-data/"
                args.train_file_source = ""
                args.test_file_source = ""

            if target in ["Real", "Syn"]:
                args.data_root_target = ""
                args.train_file_target = ""
                args.test_file_target = ""

        elif dataset in ["ROD"]:

            args.class_num = 51

            if source in ["Real", "Syn"]:

                args.data_root_source = "/work/mplanamente/ROD_0.05"

                args.train_file_source = "/work/mplanamente/SPLIT_ROD/synARID_50k-split_sync_train1.txt"
                args.val_file_source = "/work/mplanamente/SPLIT_ROD/synARID_50k-split_sync_test1.txt"
                args.test_file_source = "/work/mplanamente/SPLIT_ROD/synARID_50k-split_sync_test1.txt"

            if target in ["Real", "Syn"]:

                args.data_root_target = "/work/mplanamente/ROD_0.05"
                args.train_file_target = "/work/mplanamente/SPLIT_ROD/wrgbd_40k-split_sync.txt"

            args.val_file_target = args.val_file_source  # non abbiamo un val per questo setting
            args.test_file_target = "/work/mplanamente/SPLIT_ROD/wrgbd_40k-split_sync.txt"

    elif os.environ["HOME"].split("/")[-1] == "gberton": #Franklin
        print("Franklin")
        if dataset in ["caltech101", "ncaltech101"]:
            if source in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_source = "..."
                args.train_file_source = "..."
                args.val_file_source = "..."
                args.test_file_source = "..."

            if target in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_target = "..."
                args.train_file_target = "..."
                args.val_file_target = "..."
                args.test_file_target = "..."

        elif dataset in ["cifar10", "cifar10dvs"]:
            if source in ["Real", "Syn"]:
                args.data_root_source = "/home/chiarap/event-data/"
                args.train_file_source = ""
                args.test_file_source = ""

            if target in ["Real", "Syn"]:
                args.data_root_target = ""
                args.train_file_target = ""
                args.test_file_target = ""

        elif dataset in ["ROD"]:

            args.class_num = 51

            if source in ["Real", "Syn"]:

                args.data_root_source = "/work/mplanamente/shared/NROD"

                args.train_file_source = "/work/mplanamente/shared/SPLIT_ROD/synARID_50k-split_sync_train1.txt"
                args.val_file_source = "/work/mplanamente/shared/SPLIT_ROD/synARID_50k-split_sync_test1.txt"
                args.test_file_source = "/work/mplanamente/shared/SPLIT_ROD/synARID_50k-split_sync_test1.txt"

            if target in ["Real", "Syn"]:

                args.data_root_target = "/work/mplanamente/shared/NROD"
                if args.NPZ:
                    args.data_root_target = "/work/mplanamente/shared/ROD_NPZ"

                args.train_file_target = "/work/mplanamente/shared/SPLIT_ROD/wrgbd_40k-split_sync.txt"

            args.val_file_target = args.val_file_source  # non abbiamo un val per questo setting
            args.test_file_target = "/work/mplanamente/shared/SPLIT_ROD/wrgbd_40k-split_sync.txt"



    elif os.environ["HOME"].split("/")[-1] == "cmasone":  # Franklin
        print("Franklin")
        if dataset in ["caltech101", "ncaltech101"]:
            if source in ["Real", "Syn", "Sim"]:  # todo da fixare
                args.data_root_source = "..."
                args.train_file_source = "..."
                args.val_file_source = "..."
                args.test_file_source = "..."

            if target in ["Real", "Syn", "Sim"]:  # todo da fixare
                args.data_root_target = "..."
                args.train_file_target = "..."
                args.val_file_target = "..."
                args.test_file_target = "..."

        elif dataset in ["cifar10", "cifar10dvs"]:
            if source in ["Real", "Syn"]:
                args.data_root_source = "/home/chiarap/event-data/"
                args.train_file_source = ""
                args.test_file_source = ""

            if target in ["Real", "Syn"]:
                args.data_root_target = ""
                args.train_file_target = ""
                args.test_file_target = ""

        elif dataset in ["ROD"]:

            args.class_num = 51

            if source in ["Real", "Syn"]:
                args.data_root_source = "/work/mplanamente/shared/NROD"

                args.train_file_source = "/work/mplanamente/shared/SPLIT_ROD/synARID_50k-split_sync_train1.txt"
                args.val_file_source = "/work/mplanamente/shared/SPLIT_ROD/synARID_50k-split_sync_test1.txt"
                args.test_file_source = "/work/mplanamente/shared/SPLIT_ROD/synARID_50k-split_sync_test1.txt"

            if target in ["Real", "Syn"]:

                args.data_root_target = "/work/mplanamente/shared/NROD"
                if args.NPZ:
                    args.data_root_target = "/work/mplanamente/shared/ROD_NPZ"

                args.train_file_target = "/work/mplanamente/shared/SPLIT_ROD/wrgbd_40k-split_sync.txt"

            args.val_file_target = args.val_file_source  # non abbiamo un val per questo setting
            args.test_file_target = "/work/mplanamente/shared/SPLIT_ROD/wrgbd_40k-split_sync.txt"




    else: #cluster milano
        if dataset in ["caltech101", "ncaltech101"]:
            if source in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_source = "/home/mirco/CALTECH101/"
                args.train_file_source = "/home/mirco/CALTECH101/CALTECH101_split/train.txt"
                args.val_file_source = "/home/mirco/CALTECH101/CALTECH101_split/val.txt"
                args.test_file_source = "/home/mirco/CALTECH101/CALTECH101_split/test.txt"

            if target in ["Real", "Syn", "Sim"]: #todo da fixare
                args.data_root_target = "/home/mirco/CALTECH101/"
                args.train_file_target = "/home/mirco/CALTECH101/CALTECH101_split/train.txt"
                args.val_file_target = "/home/mirco/CALTECH101/CALTECH101_split/val.txt"
                args.test_file_target = "/home/mirco/CALTECH101/CALTECH101_split/test.txt"


        elif dataset in ["ROD"]:

            args.class_num = 51

            if source in ["Real", "Syn"]:
                args.data_root_source = "/home/mirco/data/ROD"
                if args.NPZ:
                    args.data_root_source = "/home/mirco/N_ROD" #modificata per N_rod

                args.train_file_source = "/home/mirco/data/ROD/SPLIT_ROD/synARID_50k-split_sync_train1.txt"
                args.val_file_source = "/home/mirco/data/ROD/SPLIT_ROD/synARID_50k-split_sync_test1.txt"
                args.test_file_source = "/home/mirco/data/ROD/SPLIT_ROD/synARID_50k-split_sync_test1.txt"

            if target in ["Real", "Syn"]:
                args.data_root_target = "/home/mirco/data/ROD"
                if args.NPZ:
                    args.data_root_target = "/home/mirco/N_ROD"#modificata per N_rod

                args.train_file_target = "/home/mirco/data/ROD/SPLIT_ROD/wrgbd_40k-split_sync.txt"

            args.val_file_target = args.val_file_source  # non abbiamo un val per questo setting
            args.test_file_target = "/home/mirco/data/ROD/SPLIT_ROD/wrgbd_40k-split_sync.txt"


def map_to_device(device, data):
    if isinstance(data, (torch.Tensor, nn.Module)):
        return data.to(device)
    elif isinstance(data, tuple):
        return tuple(map_to_device(device, d) for d in data)
    elif isinstance(data, list):
        return list(map_to_device(device, d) for d in data)
    elif isinstance(data, dict):
        return {k: map_to_device(device, v) for k, v in data.items()}
    else:
        return data


np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_no_dict_grouping(batch):
    """
    Same as the default_collate, but the dictionaries are not grouped by key.
    The original behavior for a dict type is:
        - [{sample1_dict}, {sample2_dict}]
        -> {key1: collate([sample1_dict[key1], sample2_dict[key1]])}
    Here instead we do:
        - [{sample1_dict}, {sample2_dict}]
        -> [{key1: sample1_dict[key1],...},
            {key1: sample2_dict[key1],...}]
    """

    # Same as default_collate
    # =======================
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype))

            return collate_no_dict_grouping([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_no_dict_grouping(samples)
                           for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_no_dict_grouping(samples) for samples in transposed]

    # The only mod
    # ============
    elif isinstance(elem, container_abcs.Mapping):
        # Originally:
        # return {key: default_collate([d[key] for d in batch]) for key in elem}
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_pad_events(batch):

    if isinstance(batch[0], tuple) \
            and isinstance(batch[0][0], dict) \
            and 'events' in batch[0][0]:

        events_lens = []
        for i, sample in enumerate(batch):
            events_lens.append(sample[0]['events'].shape[0])

        max_length = max(events_lens)

        new_batch = []
        for sample in batch:
            ev = sample[0]['events']
            ln = ev.shape[0]

            ev = torch.as_tensor(np.pad(
                ev, ((0, max_length - ln), (0, 0)),
                mode='constant', constant_values=0))
            new_batch.append(((ev, ln), *sample[1:]))
        batch = new_batch

    return collate_no_dict_grouping(batch)


'''
def collate_pad_events(batch):

    if isinstance(batch[0], tuple) \
            and isinstance(batch[0][0], dict) \
            and 'events' in batch[0][0]:

        events_lens = []
        print(batch[0])
        for i, (b, _) in enumerate(batch):
            events_lens.append(b['events'].shape[0])

        max_length = max(events_lens)

        new_batch = []
        for (b, label) in batch:
            ev = b['events']
            ln = ev.shape[0]

            ev = torch.as_tensor(np.pad(
                ev, ((0, max_length - ln), (0, 0)),
                mode='constant', constant_values=0))
            new_batch.append(((ev, ln), label))
        batch = new_batch

    return collate_no_dict_grouping(batch)

'''