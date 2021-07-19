import torch
from torch import nn

from omegaconf import OmegaConf
from evrepr.representations import get_representation
from spatialTransforms_torch import Compose


class EvReprHead(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.evrepr = None
        self.trainable = False

        if args.evrepr is not None:
            self.trainable = args.evrepr_trainable
            self.repr_crop = args.evrepr_crop
            self.repr_frame_size = args.evrepr_frame_size

            # If "EventVolume" from the CLI, we will ask for the
            # EventVolumeRepresentation class
            repr_name = args.evrepr + "Representation"
            cfg = OmegaConf.create({
                "name": repr_name,
                "args": {"frame_size": self.repr_frame_size,
                         # In our config we have "--eventvolume_arg" as
                         # argument but we need to provide only "arg" for the
                         # factory method
                         **{key.replace(args.evrepr.lower() + "_", ""): value
                            for key, value in vars(args).items()
                            # Filter only args of the requested repr
                            if key.startswith(args.evrepr.lower())}
                         }
            })

            # We ask the factory method to build the representation
            self.evrepr = get_representation(cfg, input_is_torch=True)

    def forward(self, inputs, rot=None, transform=None):
        assert transform is None or isinstance(transform, Compose)

        # If we don't have a runtime evrepr at all
        if self.evrepr is None:
            return inputs

        # We have an evrepr, but the current modality is already an image
        # NOTE: We detect events over images by checking if inputs is a 2
        # elements tuple containing as first value a tensor with 3 axis, the
        # last one having 4 values (i.e., events.shape = [B, Tmax, 4])
        # In case of images inputs is already a [N, C, H, W] tensor
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2 or \
                inputs[0].ndim != 3 or inputs[0].shape[-1] != 4:
            return inputs

        with torch.set_grad_enabled(self.trainable):
            # Extract the event representations
            reprs, rois, _ = self.evrepr(inputs)
            # Optionally perform transformations on the
            # extracted representations
            if transform is not None:
                # Apply the transformations
                roi = rois if self.repr_crop else None
                reprs = transform(reprs, rot=rot, roi=roi)

            return reprs
