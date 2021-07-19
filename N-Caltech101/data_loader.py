import os
import random

from PIL import Image
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (check_integrity,
                                        download_and_extract_archive)

from file_readers import get_reader


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def take_label(path_label, args):
    all_labels = {}
    count = 0

    for dir_user in sorted(os.listdir(path_label)):
        if (dir_user != "BACKGROUND_Google") or args.GB_class:
            all_labels[dir_user] = count
            count = count + 1
    return all_labels


def make_dataset(root, path_txt, modality, args, type_of_file):
    images = []

    all_labels = take_label(root,args)
    f = open(path_txt, "r")
    samples = f.readline()
    while samples != None and samples != "":
        label_sample = samples.strip().split('/')[0]

        ############################################
        ## to avoid the class  "BACKGROUND_Google" #
        ############################################
        if label_sample == "BACKGROUND_Google" and not args.GB_class:
            samples = f.readline()
            continue

        path_samples = root + "/" + samples
        #import pdb; pdb.set_trace()
        path_samples = path_samples.replace(".bin", type_of_file)
        path_samples = path_samples.replace("\n", "")

        item = (path_samples, all_labels[label_sample])
        images.append(item)

        samples = f.readline()

    return images



def get_relative_rotation(rgb_rot, event_rot):
    rel_rot = rgb_rot - event_rot
    if rel_rot < 0:
        rel_rot += 4
    assert rel_rot in range(4)
    return rel_rot



class CIFAR10_dvs(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            ds_name= "",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False, do_rot = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []


        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        print("Path", path)
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        rot = None
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.transform(img,rot)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

class Caltech101(Dataset):

    def __init__(self, root: str, path_txt: str,
                 transform: Optional[Callable] = None,
                 do_rot=False, args=None,
                 train=True,  # TODO(marco) this could be removed
                 isSource=True):

        self.modality = args.modality
        self.transforms = transform
        self.args = args
        self.multimodal = args.multimodal
        self.do_rot = do_rot
        self.train = train
        self.type_data = args.source if isSource else args.target
        self.data_format = args.source_data_format if isSource else args.target_data_format

        self.type_of_file = self.get_type_of_file(self.data_format)
        self.reader = get_reader(self.type_of_file, isSource, args)


        self.root = os.path.join(
            root,
            self.type_data + "_" + args.dataset + "_" +
            (args.modality if self.reader.is_image else self.data_format))
        self.images = make_dataset(
            self.root, path_txt=path_txt,
            modality=args.modality,
            args=args,
            type_of_file=self.type_of_file)

        print("Num_Samples --> ", len(self.images))

    def get_type_of_file(self, modality):

        if modality == "rgb":
            type_of_file = ".jpg"
        elif modality.startswith("event_"):
            type_of_file = modality.replace("event_", ".")
            type_of_file = type_of_file.replace(".images", "/images/")
        elif modality in ['evrepr']:
            type_of_file = ".npy"
        else:
            raise ValueError('Unknown FORMAT file {}. Known format '
                             'are jpg, bin, npy'.format(modality))
        return type_of_file

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path_img, target = self.images[index]


        img = self.reader(path_img)

        self.transforms.randomize_parameters()

        if self.args.weight_rot > 0 and self.do_rot:
            #if rotation-ss is actived
            rot = random.choice([0, 1, 2, 3])
            if self.reader.is_image:
                img = self.transforms(img, rot)
            return img, target, rot

        if self.reader.is_image:
            img = self.transforms(img)

        return img, target

    def __len__(self) -> int:
        return len(self.images)

