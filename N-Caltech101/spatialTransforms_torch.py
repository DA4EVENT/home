import numbers
import inspect
import collections
from PIL import Image

import torch
from torch import nn
from torchvision import ops
from torch.nn import functional as F
from torchvision.transforms import functional as VF


def get_torch_transforms(transform_compose):
    """
    Given a standard Compose([transforms..]) object, it transforms it
    into a Compose of this class, containing the same transforms but
    implemented in pytorch
    """

    torch_transforms = []
    for tr in transform_compose.transforms:
        class_name = tr.__class__.__name__
        if class_name == "ToTensor":
            continue
        # Retrieve the class in the torch implementation
        torch_class = globals()[class_name]

        # Get the arguments we need to create the torch class
        args = inspect.getfullargspec(torch_class.__init__).args
        # For each argument, use the same value used in the given transform
        args_and_values = {arg: getattr(tr, arg)
                           for arg in args if arg != 'self'}
        # Create the class with the inferred arguments
        torch_transforms.append(torch_class(**args_and_values))

    return Compose(torch_transforms)


def crop_boxes(images, boxes):

    # If all boxes are the same, we crop images in batch
    # images.shape = [B, C, H, W], boxes.shape = [B, (x0, y0, x1, y1)]
    if bool(torch.all(boxes.min(0).values == boxes.max(0).values)):
        box = boxes[0]
        new_images = images[:, :, box[1]:box[3] + 1, box[0]:box[2] + 1]
    else:
        # Boxes are different, we need to perform a crop at a time

        new_images = []
        for im, box in zip(images, boxes):
            new_images.append(im[:, box[1]:box[3] + 1, box[0]:box[2] + 1])
        new_images = torch.stack(new_images, dim=0)

    return new_images


class Compose(nn.Module):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, tensor, rot=None, roi=None):
        for t in self.transforms:
            tensor, rot, roi = t(tensor, rot, roi)
        return tensor


class Normalize(nn.Module):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respectively.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respectively.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        assert len(mean) == len(std)

    def forward(self, tensor, rot=None, roi=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        #print("QUI",tensor.shape[1])
        assert len(self.mean) == tensor.shape[1]

        mean = tensor.new_tensor(self.mean).reshape(1, -1, 1, 1)
        std = tensor.new_tensor(self.std).reshape(1, -1, 1, 1)

        tensor = (tensor - mean) / std
        return tensor, rot, roi


class Scale(nn.Module):
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
        super().__init__()
        assert isinstance(size, int) or \
               (isinstance(size, collections.Iterable) and
                len(size) == 2)
        self.size = size
        self.interpolation = {Image.BILINEAR: "bilinear",
                              Image.NEAREST: "nearest",
                              Image.BICUBIC: "bicubic"
                              }[interpolation]

    def forward(self, img, rot=None, roi=None):
        device = img.device
        new_img = []
        new_roi = []

        for sample_img, sample_roi in zip(img, roi):
            img_cropped = sample_img[:, sample_roi[1]:sample_roi[3] + 1,
                          sample_roi[0]:sample_roi[2] + 1]
            _, h, w = img_cropped.shape

            if (w <= h and w == self.size) or (h <= w and h == self.size):
                new_img.append(img_cropped)
            else:
                if w < h:
                    ow = self.size
                    oh = int(self.size * h / w)
                else:
                    oh = self.size
                    ow = int(self.size * w / h)

                print(self.interpolation)
                new_img.append(F.interpolate(img_cropped.unsqueeze(0), (oh, ow),
                                             mode=self.interpolation,
                                             align_corners=False)[0])

            new_roi.append(torch.tensor([0, 0, ow - 1, oh - 1],
                                        device=device))

        maxh = max([im.shape[1] for im in new_img])
        maxw = max([im.shape[2] for im in new_img])

        new_img = torch.stack([F.pad(im, (0, maxw - im.shape[2],
                                          0, maxh - im.shape[1],
                                          0, 0))
                               for im in new_img], dim=0)
        new_roi = torch.stack(new_roi, dim=0)

        return new_img, rot, new_roi


class Scale_ReplicateBorder(Scale):
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
        super().__init__(size, interpolation)
        self.mode = "replicate"

    def forward(self, img, rot=None, roi=None):
        B, C, H, W = img.shape
        device = img.device
        new_img = []

        for sample_img, sample_roi in zip(img, roi):
            img_cropped = sample_img[:, sample_roi[1]:sample_roi[3] + 1,
                          sample_roi[0]:sample_roi[2] + 1]
            _, h, w = img_cropped.shape

            if (w <= h and w == self.size) or (h <= w and h == self.size):
                new_img.append(img_cropped)
            else:
                if w > h:
                    ow = self.size
                    oh = int(self.size * h / w)
                else:
                    oh = self.size
                    ow = int(self.size * w / h)
                new_img.append(F.interpolate(img_cropped.unsqueeze(0), (oh, ow),
                                             mode=self.interpolation,
                                             align_corners=False)[0])

        new_img = torch.stack([F.pad(
            im.unsqueeze(0),
            ((self.size - im.shape[2]) // 2,
             (self.size - im.shape[2]) - (self.size - im.shape[2]) // 2,
             (self.size - im.shape[1]) // 2,
             (self.size - im.shape[1]) - (self.size - im.shape[1]) // 2),
            mode=self.mode)[0] for im in new_img], dim=0)

        new_roi = torch.tensor([0, 0, self.size - 1, self.size - 1],
                               device=device).repeat(B, 1)

        return new_img, rot, new_roi


class CenterCrop(nn.Module):

    def __init__(self, size):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def forward(self, img, rot=None, roi=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W)
        Returns:
            Tensor: Normalized image.
        """

        B, C, H, W = img.shape
        device = img.device
        th, tw = self.size

        if roi is None:
            # If no ROI is given, we consider no padding
            # is in place in the input images
            boxes = torch.tensor([int(round((W - tw) / 2.)),  # x1
                                  int(round((H - th) / 2.)),  # y1
                                  int(round((W - tw) / 2.)) + tw - 1,   # x2
                                  int(round((H - th) / 2.)) + th - 1],  # y2
                                 device=device, dtype=torch.int64)
            boxes = boxes.repeat(B, 1)
            # boxes = torch.cat([torch.arange(B, device=device).reshape(B, 1),
            #                    boxes.repeat(B, 1)],  dim=-1)
        else:
            ws = roi[:, 2] - roi[:, 0] + 1
            hs = roi[:, 3] - roi[:, 1] + 1
            roi_w_center = roi[:, 0] + ws // 2
            roi_h_center = roi[:, 1] + hs // 2
            boxes = torch.stack([#torch.arange(B, device=device),
                roi_w_center - tw // 2,
                roi_h_center - th // 2,
                roi_w_center - tw // 2 + tw - 1,
                roi_h_center - th // 2 + th - 1], dim=-1)

        cropped = crop_boxes(img, boxes)
        # cropped = ops.roi_pool(img, boxes.to(img.dtype), (th, tw))
        # Now the ROIs are all the same, covering all the frame
        roi = img.new_tensor([0, 0, tw - 1, th - 1]).repeat(B, 1)

        return cropped, rot, roi


class RandomCrop(nn.Module):

    def __init__(self, size):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def forward(self, img, rot=None, roi=None):

        B, C, H, W = img.shape
        device = img.device
        th, tw = self.size

        if roi is None:
            # If no ROI is given, we consider no padding
            # is in place in the input images
            x1 = torch.randint(0, W - tw + 1, [B], device=device)
            y1 = torch.randint(0, H - th + 1, [B], device=device)
            boxes = torch.stack([# torch.arange(B, device=device),
                x1, y1, x1 + tw - 1, y1 + th - 1], dim=-1)
        else:
            boxes = []
            for b, (rx1, ry1, rx2, ry2) in enumerate(roi):
                # b = img.new_tensor(b)
                # Note: the last +1 is because randint samples from [low, high)
                x1 = torch.randint(rx1, rx2 + 1 - tw + 1, (1,), device=device)
                y1 = torch.randint(ry1, ry2 + 1 - th + 1, (1,), device=device)
                boxes.append(torch.cat([x1, y1, x1 + tw - 1, y1 + th - 1]))
                # boxes.append(torch.cat([b, x1, y1, x1 + tw, y1 + th]))
            boxes = torch.stack(boxes, dim=0)

        # cropped = ops.roi_pool(img, boxes, (th, tw))
        cropped = crop_boxes(img, boxes)
        # Now the ROIs are all the same, covering all the frame
        roi = img.new_tensor([0, 0, tw-1, th-1]).repeat(B, 1)

        return cropped, rot, roi


class Rotation(nn.Module):

    def __init__(self):
        super().__init__()

    def group_rot(self, rot):
        groups = [[] for _ in range(4)]
        for i, r in enumerate(rot):
            groups[int(r)].append(i)
        return groups

    def roi_rot(self, roi, r, shape):
        H, W = shape

        if r == 0:
            return roi
        elif r == 1:
            x_bck = roi[:, [0, 2]].clone()
            roi[:, [0, 2]] = roi[:, [1, 3]]
            roi[:, [1, 3]] = W - 1 - x_bck
        elif r == 2:
            roi[:, [0, 2]] = W - 1 - roi[:, [0, 2]]
            roi[:, [1, 3]] = H - 1 - roi[:, [1, 3]]
        elif r == 3:
            x_bck = roi[:, [0, 2]].clone()
            roi[:, [0, 2]] = H - 1 - roi[:, [1, 3]]
            roi[:, [1, 3]] = x_bck

        # Make sure (x0,y0) in the top left and (x1, y1) in bottom right
        roi[:, [0, 2]] = torch.sort(roi[:, [0, 2]], dim=-1).values
        roi[:, [1, 3]] = torch.sort(roi[:, [1, 3]], dim=-1).values

        return roi

    def forward(self, img, rot=None, roi=None):

        B, C, H, W = img.shape
        assert H == W

        if rot is None:
            rot = torch.randint(0, self.angles.shape[0], [B])
        # Get a list [[rot0_sample_ids], [rot1_sample_ids], ...]
        groups = self.group_rot(rot)

        for r, sample_ids in enumerate(groups):
            img[sample_ids] = torch.rot90(img[sample_ids], k=r, dims=(2, 3))
            roi[sample_ids] = self.roi_rot(roi[sample_ids], r, (H, W))

        return img, rot, roi


class RandomHorizontalFlip(nn.Module):

    def forward(self, img, rot=None, roi=None):

        B, C, H, W = img.shape
        device = img.device

        flip_mask = torch.rand(B, device=device) < 0.5
        # Flip the image
        img[flip_mask] = torch.flip(img[flip_mask], dims=(-1,))
        # And the ROIs as well (we just need to flip the x coords)
        x0_bck = roi[flip_mask, 0].clone()
        roi[flip_mask, 0] = W - 1 - roi[flip_mask, 2]
        roi[flip_mask, 2] = W - 1 - x0_bck

        return img, rot, roi


if __name__ == "__main__":
    from skimage import data
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    # Visual debug code

    images = data.lfw_subset()

    batch = images[:3].reshape(3, 1, 25, 25)
    batch = torch.tensor(batch)

    # roi = torch.tensor([[0, 0, 11, 11],
    #                     [13, 13, 24, 24]])
    # transform = CenterCrop(10)
    # transform = RandomHorizontalFlip()

    # roi = torch.tensor([[0, 0, 9, 9],
    #                     [0, 13, 24, 24],
    #                     [0, 13, 11, 24]])
    # transform = RandomCrop(10)

    # roi = torch.tensor([[0, 0, 20, 10],
    #                     [0, 0, 20, 10],
    #                     [0, 0, 20, 10]])
    # transform = Rotation()
    # rot = torch.tensor([1, 2, 3])

    roi = torch.tensor([[0, 0, 20, 10],
                        [0, 5, 10, 24],
                        [0, 0, 10, 10]])
    # transform = Scale(25)
    transform = Scale_ReplicateBorder(25)
    rot = None

    batch_tr, rot_tr, roi_tr = transform(batch.clone(), rot=rot, roi=roi.clone())
    assert (rot is None and rot_tr is None) or torch.all(rot_tr == rot)

    roi = [None] * 3 if roi is None else roi
    for im, r, rt, tr in zip(batch, roi, roi_tr, batch_tr):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ax = axes.ravel()

        ax[0].imshow(im[0].detach().numpy())
        if r is not None:
            ax[0].add_patch(Rectangle((int(r[0]), int(r[1])),
                                      int(r[2] - r[0]), int(r[3] - r[1]),
                                      edgecolor='red', fill=False, lw=1))
        print("original img", im[0].shape)
        if r is not None:
            print("original roi", int(r[2] - r[0] + 1), int(r[3] - r[1] + 1))

        ax[1].imshow(tr[0].detach().numpy())
        ax[1].add_patch(Rectangle((int(rt[0]), int(rt[1])),
                                  int(rt[2] - rt[0]), int(rt[3] - rt[1]),
                                  edgecolor='red', fill=False, lw=1))
        print("transf img", tr[0].shape)
        print("transf roi", int(rt[2] - rt[0] + 1), int(rt[3] - rt[1] + 1))

        fig.tight_layout()
        plt.show()
