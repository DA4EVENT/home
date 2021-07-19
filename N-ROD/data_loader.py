import os
import random

import os.path
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
from file_readers import get_reader

def make_dataset_ROD(root, path_txt, modality, type_of_file, type_data):
    images = []

    labels_txt = open(path_txt, "r")
    for line in labels_txt:
        label_sample = int(line.strip().split(' ')[1])
        sample = line.strip().split(' ')[0]
        path = os.path.join(root, sample)

        if type_data == "Real": #todo deve essere adattato una volta che abbiamo i dati completi
            name = "depthcrop" if modality == "depth" else "crop"
            path_samples = path.replace('***', name) #todo da capire se va bene
            path_samples = path_samples.replace('???', modality)
        else:
            path_samples = path.replace("***", modality)


        path_samples = path_samples.replace("\n", "")
        path_samples = path_samples.replace(".png", type_of_file)

        item = (path_samples, label_sample)
        images.append(item)
    return images


def make_dataset_ROD_MultiModal(root, path_txt, modality, type_of_file_1, type_of_file_2, type_data):
    images = []

    ### Multi Modal
    modality_1, modality_2 = modality.split("-") # RGB and Voxel

    labels_txt = open(path_txt, "r")
    for line in labels_txt:
        label_sample = int(line.strip().split(' ')[1])
        sample = line.strip().split(' ')[0]
        path = os.path.join(root, sample)
        if type_data == "Real":
            name = "depthcrop" if modality_1 == "depth" else "crop"

            path_samples_1 = path.replace('***', name)
            path_samples_1 = path_samples_1.replace('???', modality_1)

            name = "depthcrop" if modality_2 == "depth" else "crop"
            path_samples_2 = path.replace('***', name)
            path_samples_2 = path_samples_2.replace('???', modality_2)
        else:
            path_samples_1 = path.replace("***", modality_1)
            path_samples_2 = path.replace("***", modality_2)

        path_samples_1 = path_samples_1.replace("\n", "")
        path_samples_1 = path_samples_1.replace(".png", type_of_file_1)
        path_samples_2 = path_samples_2.replace("\n", "")
        path_samples_2 = path_samples_2.replace(".png", type_of_file_2)


        item = (path_samples_1, path_samples_2, label_sample)
        images.append(item)

    return images


def get_relative_rotation(rgb_rot, event_rot):
    rel_rot = rgb_rot - event_rot
    if rel_rot < 0:
        rel_rot += 4
    assert rel_rot in range(4)
    return rel_rot


###########
#   ROD   #
###########

class ROD(Dataset):

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

        if self.multimodal:
            format_1, format_2 = self.data_format.split("-")
            self.type_of_file_1 = self.get_type_of_file(format_1)
            self.type_of_file_2 = self.get_type_of_file(format_2)
            self.reader_1 = get_reader(self.type_of_file_1, isSource, args)
            self.reader_2 = get_reader(self.type_of_file_2, isSource, args)

            self.root = os.path.join(
                root,
                self.type_data + args.dataset)

            self.images = make_dataset_ROD_MultiModal(
                self.root,
                path_txt=path_txt,
                modality=args.modality if self.reader_1.is_image else self.data_format, # Rgb or Voxel
                type_of_file_1=self.type_of_file_1, #png or npy
                type_of_file_2=self.type_of_file_2, #png or npy
                type_data = self.type_data) #Real o Syn
            print("Path Modality --> ", self.root)


        else:
            self.type_of_file = self.get_type_of_file(self.data_format)
            self.reader = get_reader(self.type_of_file, isSource, args)

            #print(self.type_of_file)
            #print(self.reader.is_image)
            #self.reader.is_image = True # da togliere


            #/path/to/{Sim,Real}{dataset}
            self.root = os.path.join(
                root,
                self.type_data + args.dataset)
            self.images = make_dataset_ROD(
                self.root, path_txt=path_txt,
                modality=args.modality if self.reader.is_image else self.data_format,
                type_of_file=self.type_of_file, type_data = self.type_data)

        print("Num_Samples --> ", len(self.images))

    def get_type_of_file(self, modality):

        if (modality == "rgb") or (modality == "depth"):
            type_of_file = ".png"
        elif modality.startswith("event_"):
            type_of_file = modality.replace("event_", ".")
            type_of_file = type_of_file.replace(".images", "/images/")
        elif modality in ['evrepr']:
            type_of_file = ".npy"
        else:
            raise ValueError('Unknown FORMAT file {}. Known format '
                             'are png, bin, npy'.format(modality))
        return type_of_file

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if self.multimodal:
            path_modality1_img, path_modality2_img, target = self.images[index]

            img_mod1 = self.reader_1(path_modality1_img)
            img_mod2 = self.reader_2(path_modality2_img)

            self.transforms.randomize_parameters()

            if self.args.weight_rot > 0 and self.do_rot:
                #if rotation-ss is actived
                rot1 = random.choice([0, 1, 2, 3])
                rot2 = random.choice([0, 1, 2, 3])
                if self.reader_1.is_image:
                    img_mod1 = self.transforms(img_mod1, rot1)
                if self.reader_2.is_image:
                    img_mod2 = self.transforms(img_mod2, rot2)
                relative_rotation = get_relative_rotation(rot1, rot2)
                return img_mod1, img_mod2, target, relative_rotation, rot1, rot2

            if self.reader_1.is_image:
                img_mod1 = self.transforms(img_mod1)
            if self.reader_2.is_image:
                img_mod2 = self.transforms(img_mod2)

            return img_mod1, img_mod2, target

        else:
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
