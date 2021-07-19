import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)

    args = parser.parse_args()
    return args


def remove_padding(img):
    H, W, C = img.shape

    top_row = img[0:1, :, :]  # [1, W, C]
    mask = (img == top_row).all(-1).all(1)
    top = np.clip(np.argmin(mask) - 1, 0, H - 1)

    bottom_row = img[H-1:H, :, :]  # [1, W, C]
    mask = (img == bottom_row).all(-1).all(1)[::-1]
    bottom = np.clip(H - np.argmin(mask), 0, H - 1)

    left_col = img[:, 0:1, :]  # [H, 1, C]
    mask = (img == left_col).all(-1).all(0)
    left = np.clip(np.argmin(mask) - 1, 0, W - 1)

    right_col = img[:, W-1:W, :]  # [H, 1, C]
    mask = (img == right_col).all(-1).all(0)[::-1]
    right = np.clip(W - np.argmin(mask), 0, W - 1)

    return img[top:bottom+1, left:right+1]


def process_rgbs(data_path):
    sep = os.path.sep
    rgb_paths = glob(os.path.join(data_path, "**/rgb/*.png"), recursive=True)
    for rgb_path in tqdm(rgb_paths):
        out_path = rgb_path.replace(sep + "rgb" + sep,
                                    sep + "rgb_nopad" + sep)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = cv2.imread(rgb_path)
        img = remove_padding(img)
        cv2.imwrite(out_path, img)


if __name__ == "__main__":
    args = parse_args()
    process_rgbs(args.dataset_dir)
