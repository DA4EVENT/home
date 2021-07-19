import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from evrepr.utils.args import parse_args
from evrepr.utils.logger import setup_logger
from evrepr.data import get_loader, map_to_device
from evrepr.representations import get_representation
from evrepr.utils.visualization import postprocess_representation, pformat_dict

logger = setup_logger("evrepr.tools.extract")


def main():
    cfg = parse_args()
    assert cfg.action in ['display', 'save']
    assert cfg.action != "save" or cfg.data.to_torch is False

    ev_repr = get_representation(cfg.repr)
    ev_repr.to(torch.device(cfg.device))
    loader = get_loader(cfg.data)

    logger.info("Running with parameters: " + pformat_dict(cfg, indent=1))

    window_name = None
    if cfg.action == 'display':
        window_name = "Event Reconstruction"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with torch.no_grad():
        for sample_idx, inputs in tqdm(enumerate(loader), total=len(loader)):
            inputs = map_to_device(inputs, cfg.device)
            images, rois, _ = ev_repr(inputs)
            images = images.detach().cpu().numpy()

            for input_dict, image, roi in zip(inputs, images, rois):
                x1, y1, x2, y2 = roi

                if cfg.action == 'display':
                    if cfg.crop_repr:
                        image = image[:, y1:y2 + 1, x1:x2 + 1]
                    image = postprocess_representation(
                        image, cfg.viz.group_channels,
                        cfg.viz.clip_outliers)

                    cv2.imshow(window_name, image)
                    if not cfg.data.to_torch:
                        logger.info(input_dict['path'])
                    key = cv2.waitKey(cfg.viz.display_wait)
                    if key & 0xff == ord('q'):
                        cv2.destroyWindow(window_name)
                        return

                elif cfg.action == 'save':
                    file_path = input_dict['path']
                    rel_path = os.path.relpath(file_path, cfg.data.path)
                    new_path = os.path.join(
                        cfg.output_path,
                        os.path.splitext(rel_path)[0] + ".npy")
                    new_path = new_path.replace("/images.", ".")
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)

                    if cfg.crop_repr:
                        image = image[:, y1:y2 + 1, x1:x2 + 1]
                    np.save(new_path, image)


if __name__ == "__main__":
    main()

