import random
import math
import numbers
import collections
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """
    Composes several transforms together.
    Args:
        transforms (list of Transform objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, rot1=None,rot2=None):
        for t in self.transforms:
            img1, img2 = t(img1, img2, rot1, rot2)
        return img1, img2

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value1=255,norm_value2=1):
        self.norm_value1 = norm_value1
        self.norm_value2 = norm_value2

    def __call__(self, pic1, pic2, rot1, rot2):

        # For The Second Modality
        if isinstance(pic2, np.ndarray):
            # handle numpy array
            img2 = torch.from_numpy(pic2.transpose((2, 0, 1)))
            # backward compatibility
            img2 = img2.float().div(self.norm_value2)


        img1 = torch.ByteTensor(torch.ByteStorage.from_buffer(pic1.tobytes()))
        nchannel = len(pic1.mode)

        img1 = img1.view(pic1.size[1], pic1.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img1 = img1.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img1, torch.ByteTensor):
            img1 = img1.float().div(self.norm_value1)

        return img1, img2

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor1, tensor2, rot1, rot2):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        mean = self.mean
        std = self.std
        for t, m, s in zip(tensor1, mean, std):
            t.sub_(m).div_(s)
        for t, m, s in zip(tensor2, mean, std):
            t.sub_(m).div_(s)
        return tensor1, tensor2

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img1,  img2, rot1, rot2):

        ####################
        #   Mod 1 --> RGB  #
        ####################

        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                pass
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                img1 = img1.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                img1 = img1.resize((ow, oh), self.interpolation)
        else:
            img1 = img1.resize(self.size, self.interpolation)

        ######################
        #   Mod 2 --> Event  #
        ######################

        if isinstance(self.size, int):
            c, w, h = img2.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img1, img2
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                #img = np.resize(img, (c, ow, oh))
                img2 = resize(img2, (c, ow, oh), anti_aliasing=True)

                #return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                img2 = resize(img2, (c, ow, oh), anti_aliasing=True)
                #return img.resize((ow, oh), self.interpolation)
        else:
            return img2.resize(self.size, self.interpolation)

        return img1, img2


    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1,  img2, rot1, rot2):

        ####################
        #   Mod 1 --> RGB  #
        ####################

        w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img1 = img1.crop((x1, y1, x1 + tw, y1 + th))

        ######################
        #   Mod 2 --> Event  #
        ######################

        c, w, h = img2.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img2 = img2[:, y1:y1 + th, x1:x1 + tw]

        return img1, img2

    def randomize_parameters(self):
        pass

class RandomCrop(object):
    """Random the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.top = 0
        self.left = 0

    def __call__(self, img1,  img2, rot1, rot2):

        ####################
        #   Mod 1 --> RGB  #
        ####################

        w, h = img1.size
        th, tw = self.size

        self.top = random.randint(0, h - th)
        self.left = random.randint(0, w - tw)

        img1 = TF.crop(img1, self.top, self.left, 224, 224)

        ######################
        #   Mod 2 --> Event  #
        ######################


        return img1

    def randomize_parameters(self):
        pass


class Rotation(object):

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img, rot):
        img = TF.rotate(img, self.angles[rot])
        return img

    def randomize_parameters(self):
        pass



class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img, rot):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        self.p = random.random()
        if self.p < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()

