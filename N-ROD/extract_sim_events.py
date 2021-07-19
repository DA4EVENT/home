import os
import cv2
import esim_py
import argparse
import tempfile
import numpy as np
from tqdm import tqdm
from glob import glob
import multiprocessing


def arg_boolean(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esim_contrast_threshold', type=float, default=0.06)
    parser.add_argument('--esim_refractory_period', type=float, default=1e6)
    parser.add_argument('--esim_use_log', type=arg_boolean, default=True)
    parser.add_argument('--esim_log_eps', type=float, default=0.001)
    parser.add_argument('--esim_timestamps', type=str)

    parser.add_argument('--remove_padding', type=arg_boolean, default=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--dataset_videos', type=str)
    parser.add_argument('--dataset_crops', type=str)

    args = parser.parse_args()
    return args


def get_padding_box(img):
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

    return left, top, right, bottom


def crop_images(img_dir, dest_dir, roi):
    x1, y1, x2, y2 = roi
    img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
    for path in img_paths:
        img_name = os.path.basename(path)
        dest_path = os.path.join(dest_dir, img_name)

        img = cv2.imread(path)
        img = img[y1:y2+1, x1:x2+1]
        cv2.imwrite(dest_path, img)


def process_sample(job_args):
    crop_path, vid_path, args = job_args

    esim = esim_py.EventSimulator(
        args.esim_contrast_threshold,
        args.esim_contrast_threshold,
        args.esim_refractory_period,
        args.esim_log_eps,
        args.esim_log_eps)

    if args.remove_padding:
        # Get the original padded image dimension
        pad_crop = cv2.imread(crop_path)
        h, w, _ = pad_crop.shape
        assert h == w

        # Get the roi size before the 256x256 resize
        roi = np.array(get_padding_box(pad_crop))
        s = 256 / h

        # Resize the roi to 256x256
        roi = (roi * s).astype(np.int32)
        assert np.all(roi < 256)

        # Enlarge the roi size to frame the motion
        margin = 5  # px
        roi[[0, 1]] = roi[[0, 1]] - margin
        roi[[2, 3]] = roi[[2, 3]] + margin
        roi = np.clip(roi, 0, 255)

        # Uniform background may be present, resulting in a roi
        # not extending to one of the sides. In that case we manually
        # extend the roi to the largest size
        roiw, roih = roi[[2, 3]] - roi[[0, 1]]
        if roih > roiw:
            roi[[1, 3]] = (0, 255)
        else:
            roi[[0, 2]] = (0, 255)

        # We now crop the images in a temporary folder,
        # and generate events from that folder
        with tempfile.TemporaryDirectory() as tmp_path:
            crop_images(vid_path, tmp_path, roi)
            events = esim.generateFromFolder(tmp_path, args.esim_timestamps)

    else:
        events = esim.generateFromFolder(vid_path, args.esim_timestamps)

    np.save(os.path.dirname(vid_path) + ".npy", events)


def find_images(dataset_crops, dataset_videos):
    print("Listing dataset samples...")
    crop_paths = glob(os.path.join(dataset_crops, "*/rgb/*.png"))
    video_paths = []
    for path in crop_paths:
        # /path/to/data/cls/rgb/img.png -> cls/rgb/img
        vpath = os.path.splitext(os.path.relpath(path, dataset_crops))[0]
        # cls/rgb/img -> /path/to/video/cls/rgb_256x256/img/
        vpath = os.path.join(dataset_videos,
                             vpath.replace(os.sep + "rgb" + os.sep,
                                           os.sep + "rgb_256x256" + os.sep))
        assert os.path.exists(vpath)
        video_paths += [vpath]

    print("Done! {} images found!".format(len(crop_paths)))
    assert len(crop_paths) > 0

    return crop_paths, video_paths


def process_folder(args):

    # [(sample0_crop, sample0_vid), (sample1_crop, sample1_vid), ...]
    crop_paths, video_paths = find_images(args.dataset_crops,
                                          args.dataset_videos)

    # Builds arguments for jobs
    num_samples = len(crop_paths)
    job_args = zip(crop_paths, video_paths, [args] * num_samples)

    # Process each image independently in a different job
    with multiprocessing.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap(process_sample, job_args),
                  total=num_samples))

    # for arg in job_args:
    #     process_sample(arg)


if __name__ == "__main__":
    args = parse_args()
    process_folder(args)
