import os
import cv2
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool


ORIGINAL_RADIUS = 10
ORIGINAL_CENTER = (10, 0)
ORIGINAL_SIZE = 512
TARGET_SIZE = 128


def arg_boolean(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--fit_movement", type=arg_boolean, default=True)
    parser.add_argument("--use_symlinks", type=arg_boolean, default=True)
    parser.add_argument("--subsample_10k", type=arg_boolean, default=True)

    parser.add_argument("--trajectory_inpoints", type=int, default=4)
    parser.add_argument("--trajectory_cycles", type=int, default=6)
    parser.add_argument("--trajectory_duplicates", type=arg_boolean, default=False)

    args = parser.parse_args()
    return args


def round(v):
    return int(np.round(v))


def int_abs(v):
    return int(math.fabs(v))


def resize(img, size=TARGET_SIZE,
           interpolation=cv2.INTER_CUBIC):
    h, w, _ = img.shape
    assert h == w
    if not isinstance(size, (list, tuple)):
        size = (size, size)

    return cv2.resize(img, size, interpolation=interpolation)


def scale_radius(orig_radius=ORIGINAL_RADIUS,
                 orig_size=ORIGINAL_SIZE,
                 target_size=TARGET_SIZE):
    scale = target_size / orig_size
    return scale * orig_radius


def scale_center(orig_center=ORIGINAL_CENTER,
                 orig_size=ORIGINAL_SIZE,
                 target_size=TARGET_SIZE):
    scale = target_size / orig_size
    xc, yc = orig_center
    return scale * xc, scale * yc


def fit_size_movement(size, trajectory):
    min_x, max_x = trajectory[:, 0].min(), trajectory[:, 0].max()
    min_y, max_y = trajectory[:, 1].min(), trajectory[:, 1].max()
    border_x = max_x - min_x
    border_y = max_y - min_y

    fit_x = int(border_x) + int_abs(min_x) + size
    fit_y = int(border_y) + int_abs(min_y) + size

    return fit_x, fit_y


def get_trajectory(center=ORIGINAL_CENTER, radius=ORIGINAL_RADIUS,
                   inpoints=4, cycles=6, duplicates=True):
    xc, yc = center
    r = radius

    # The 4 points of the square centered in (xc, yc)
    p0 = (xc - r, yc)
    p1 = (xc, yc - r)
    p2 = (xc + r, yc)
    p3 = (xc, yc + r)

    # The end points of the paths
    points = [p0, p1, p2, p3, p0]
    final_points = []

    alphas = np.linspace(1, 0, 2 + inpoints)
    if duplicates is False:
        alphas = alphas[1:]

    # Interpolate points in each path
    for p in range(len(points) - 1):
        x_start, y_start = points[p]
        x_end, y_end = points[p+1]
        final_points += [(a * x_start + (1 - a) * x_end,
                          a * y_start + (1 - a) * y_end)
                         for a in alphas]

    # Replicate the path for the number of cycles
    final_points = np.array(final_points * cycles)
    return final_points


def translate_image(img, path, crop_border=True):
    h, w, c = img.shape
    n = path.shape[0]

    x_max_offset = path[:, 0].max()
    x_min_offset = path[:, 0].min()
    border_x = x_max_offset - x_min_offset

    y_max_offset = path[:, 1].max()
    y_min_offset = path[:, 1].min()
    border_y = y_max_offset - y_min_offset

    new_imgs = np.zeros((n, round(h + border_y),
                         round(w + border_x), c),
                        dtype=img.dtype)
    for i, (x, y) in enumerate(path):
        xs = x - x_min_offset
        ys = y - y_min_offset

        # Compute the affine transformation matrix
        M = np.float32([[1, 0, xs], [0, 1, ys]])
        new_img = cv2.warpAffine(img, M, (w, h))

        new_h, new_w, _ = new_img.shape
        new_imgs[i, :new_h, :new_w] = new_img

    if crop_border:
        new_imgs = new_imgs[:,
                   int(border_y):h-int_abs(y_min_offset),
                   int(border_x):w-int_abs(x_min_offset)]

    return new_imgs


def process_image(jobargs):
    img_path, trajectory, remove_subdirs, args = jobargs
    # Define the output path by maintaining the same directory
    # tree structure within output_dir as in the input_dir
    rel_path = os.path.relpath(img_path, start=args.input_dir)
    rel_path = os.path.splitext(rel_path)[0]
    out_path = os.path.join(args.output_dir, rel_path, "images")
    for rm_subdir in remove_subdirs:
        out_path = out_path.replace(rm_subdir + os.sep, "")
    os.makedirs(out_path, exist_ok=True)

    # Load the image
    img = cv2.imread(img_path)
    # Define the image size s.t. it (optionally) fit the movement
    if args.fit_movement:
        fit_size = fit_size_movement(TARGET_SIZE, trajectory)
    else:
        fit_size = TARGET_SIZE
    # Resize the image
    img = resize(img, fit_size)
    # Perform the movement
    imgs = translate_image(img, trajectory,
                           crop_border=args.fit_movement)

    # Save the "video"
    for i, img in enumerate(imgs):
        path_i = os.path.join(out_path, "image_{:04}.png".format(i))
        cv2.imwrite(path_i, img)
    # If use_symlinks = True, imgs contains only one cycle,
    # the remaining is created with symlinks
    if args.use_symlinks:
        base_i = len(imgs)
        for _ in range(args.trajectory_cycles):
            for i, img in enumerate(imgs):
                os.symlink(
                    "image_{:04}.png".format(i),
                    os.path.join(out_path, "image_{:04}.png".format(base_i + i))
                )
            base_i += len(imgs)


def save_split(paths, name, args):
    os.makedirs(args.output_dir, exist_ok=True)
    txt_path = os.path.join(args.output_dir, name + "_split.txt")
    with open(txt_path, "w") as fp:
        fp.write("".join("{}\n".format(os.path.relpath(p, start=args.input_dir))
                         for p in paths)[:-1])


def is_cifar_merged(args):
    folders = sorted(glob(os.path.join(args.input_dir, "*/")))
    if len(folders) == 2 and folders[0] == "test" and folders[1] == "train":
        return False
    if len(folders) == 10:
        return True


def sample_from_60k(args):
    print("Sampling from merged train+test folders")
    sampled_paths = []
    split_train, split_test = [], []

    class_paths = glob(os.path.join(args.input_dir, "*/"))
    assert len(class_paths) == 10

    for class_path in class_paths:
        # Read all samples in this class
        sample_paths = sorted(glob(os.path.join(class_path, "*.png")))
        assert len(sample_paths) == 6000
        # Randomly select 1000
        random.shuffle(sample_paths)
        selected_sample_paths = sample_paths[:1000]
        # Add the selected paths to those that will be converted
        sampled_paths += selected_sample_paths
        # Reserve 900 for training and 100 for testing
        split_train += selected_sample_paths[:900]
        split_test += selected_sample_paths[900:]

    # Save split paths without the file extension
    save_split([p.replace(".png", "") for p in split_train], "train", args)
    save_split([p.replace(".png", "") for p in split_test], "test", args)

    return sampled_paths


def sample_from_50k_10k(args):
    print("Sampling from split train and test folders")

    class_paths = glob(os.path.join(args.input_dir, "*/"))
    assert len(class_paths) == 2

    def sample_n_from_classes(classes_dir, take_n):
        sampled = []
        class_paths = glob(os.path.join(classes_dir, "*/"))
        assert len(class_paths) == 10

        for class_path in class_paths:
            # Read all samples in this class
            sample_paths = sorted(glob(os.path.join(class_path, "*.png")))
            # Randomly select 'take_n' samples
            random.shuffle(sample_paths)
            sampled += sample_paths[:take_n]

        return sampled

    split_train = sample_n_from_classes(
        os.path.join(args.input_dir, "train"), take_n=900)
    split_test = sample_n_from_classes(
        os.path.join(args.input_dir, "test"), take_n=100)

    assert len(split_train) == 9000 and len(split_test) == 1000
    # Save paths without the /test/ and /train/ subdirs and without extensions
    save_split([p.replace("train" + os.sep, "").replace(".png", "")
                for p in split_train], "train", args)
    save_split([p.replace("test" + os.sep, "").replace(".png", "")
                for p in split_test], "test", args)

    sampled_paths = split_train + split_test

    return sampled_paths


def sample_all(args):
    print("Sampling all samples")

    class_paths = glob(os.path.join(args.input_dir, "*/"))
    assert len(class_paths) == 2

    split_train = glob(os.path.join(args.input_dir, "train/**/*.png"),
                       recursive=True)
    split_test = glob(os.path.join(args.input_dir, "test/**/*.png"),
                      recursive=True)
    assert len(split_train) == 50000 and len(split_test) == 10000

    # Save paths without the /test/ and /train/ subdirs and without extensions
    save_split([p.replace("train" + os.sep, "").replace(".png", "")
                for p in split_train], "train", args)
    save_split([p.replace("test" + os.sep, "").replace(".png", "")
                for p in split_test], "test", args)

    sampled_paths = split_train + split_test

    return sampled_paths


def process_folder(args):
    random.seed(42)
    np.random.seed(42)

    print("Searching images...")
    is_merged = is_cifar_merged(args)
    if args.subsample_10k:
        if is_merged:
            sampled_paths = sample_from_60k(args)
        else:
            sampled_paths = sample_from_50k_10k(args)
    else:
        sampled_paths = sample_all(args)
    num_images = len(sampled_paths)
    print("Done! {} images randomly selected!".format(num_images))

    radius = scale_radius(orig_radius=ORIGINAL_RADIUS,
                          orig_size=ORIGINAL_SIZE,
                          target_size=TARGET_SIZE)
    center = scale_center(orig_center=ORIGINAL_CENTER,
                          orig_size=ORIGINAL_SIZE,
                          target_size=TARGET_SIZE)
    # If use_symlinks = False, we explicitly replicate the trajectory,
    # this will generate and save duplicate images.
    # If it is true, we avoid replicating the trajectory, we will later
    # create symlinks to simulate the cycle
    cycles = 1 if args.use_symlinks else args.trajectory_cycles
    trajectory = get_trajectory(center=center, radius=radius,
                                inpoints=args.trajectory_inpoints,
                                cycles=cycles,
                                duplicates=args.trajectory_duplicates)

    # Builds the args for each job
    trajectory_jobargs = [trajectory] * num_images
    args_jobargs = [args] * num_images
    rm_subdirs_jobargs = [[] if is_merged else ['train', 'test']] * num_images
    jobargs = zip(sampled_paths, trajectory_jobargs,
                  rm_subdirs_jobargs, args_jobargs)

    # Process each image independently in a different job
    with Pool(args.num_workers) as pool:
        list(tqdm(pool.imap(process_image, jobargs),
                  total=num_images))

    # for img_path in tqdm(images):
    #     process_image((img_path, trajectory, args))


if __name__ == "__main__":

    args = parse_args()
    process_folder(args)

