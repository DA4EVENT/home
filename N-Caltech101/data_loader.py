import os
import random

import os.path
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset

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



class Caltech101(Dataset):

    def __init__(self, root: str, path_txt: str,
                 transform: Optional[Callable] = None,
                 do_rot=False, args=None,
                 train=True,
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

