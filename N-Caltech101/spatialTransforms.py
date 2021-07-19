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
from skimage.transform import resize


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

    def __call__(self, img, rot=None):
        for t in self.transforms:
            img = t(img, rot)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value
        self.norm_value_event = 1

    def __call__(self, pic, rot):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.copy())
            # backward compatibility
            return img.float().div(self.norm_value_event)


        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

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

    def __call__(self, tensor, rot):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        mean = self.mean
        std = self.std
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

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

    def __call__(self, img, rot):

        ######################
        #   Mod 2 --> Event  #
        ######################

        if isinstance(img, np.ndarray) and isinstance(self.size, int):
            c, h, w = img.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                # img = np.resize(img, (c, ow, oh))
                img = resize(img, (c, oh, ow), anti_aliasing=True)
                return img
            else:
                oh = self.size
                ow = int(self.size * w / h)
                img = resize(img, (c, oh, ow), anti_aliasing=True)
                return img

        ####################
        #   Mod 1 --> RGB  #
        ####################

        elif isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

        return img

    def randomize_parameters(self):
        pass

class Scale_ReplicateBorder(object):
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

    def __call__(self, img, rot):
        #scale to 256 the longer side and scale proportionally the other one and
        #replicate the border of short side in order to obtain 256x256
        #ONLY FOR Event

        ######################
        #   Mod 2 --> Event  #
        ######################

        if isinstance(img, np.ndarray) and isinstance(self.size, int):
            c, h, w = img.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                img =  img
            if w > h:
                axis = 1
                ow = self.size
                oh = int(self.size * h / w)
                dim = np.ones(oh, dtype=int)
                n_repeat = (( ow - oh ) // 2) + 1
                img = resize(img, (c, oh, ow), anti_aliasing=True)
            else:
                axis = 2
                oh = self.size
                ow = int(self.size * w / h)
                dim = np.ones(ow, dtype=int)
                n_repeat = (( oh - ow ) // 2) + 1
                img = resize(img, (c, oh, ow), anti_aliasing=True)

            dim[0] = n_repeat
            dim[len(dim)-1] = n_repeat
            img = np.repeat(img, dim, axis=axis)
            _, n_h, n_w = img.shape

            return img

        ####################
        #   Mod 1 --> RGB  #
        ####################

        elif isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

        return img

    def randomize_parameters(self):
        pass

class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, rot):

        #EVENT
        if isinstance(img, np.ndarray):
            c, h, w = img.shape
            th, tw = self.size
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return img[:, y1:y1 + th, x1:x1 + tw]

        #RGB
        else:
            w, h = img.size
            th, tw = self.size
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass

class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.top = 0
        self.left = 0

    def __call__(self, img, rot):

        #EVENT
        if isinstance(img, np.ndarray):
            c, h, w = img.shape
            th, tw = self.size
            self.top = random.randint(0, h - th)
            self.left = random.randint(0, w - tw)
            img = img[:, self.top : self.top+th, self.left:self.left + tw]
            return img

        #RGB
        else:
            w, h = img.size
            th, tw = self.size

            self.top = random.randint(0, h - th)
            self.left = random.randint(0, w - tw)

            img = TF.crop(img, self.top, self.left, 224, 224)

        return img

    def randomize_parameters(self):
        pass


class Rotation(object):

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img, rot):

        # EVENT
        if isinstance(img, np.ndarray):
            img = np.rot90(img, k=rot, axes=(1, 2))

        #RGB
        else:
            img = TF.rotate(img, self.angles[rot])

        return img

    def randomize_parameters(self):
        pass

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img, rot):

        if self.p < 0.5:
            #EVENT
            if isinstance(img, np.ndarray):
                img = np.flip(img, axis=-1)

            else:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return img

    def randomize_parameters(self):
        self.p = random.random()

