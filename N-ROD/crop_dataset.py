import os
import yaml
import argparse
from PIL import Image
from glob import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--bboxes_margin', type=int, default=10)

    args = parser.parse_args()
    return args


def crop_bbox(png_path, bbox, margin):
    img = Image.open(png_path, 'r')
    w, h = img.size
    y1, x1, y2, x2 = bbox

    return img.crop((max(0, x1 - margin), max(0, y1 - margin),
                     min(w, x2 + margin), min(h, y2 + margin)))


def process_scene(scene_path, output_path, margin):
    yaml_path = os.path.dirname(scene_path) + "gt.yaml"
    depths_path = glob(os.path.join(scene_path, "depth/*.png"))
    parts_path = glob(os.path.join(scene_path, "part/*.png"))
    seg_path = os.path.join(scene_path, "mask.png")
    rgb_path = os.path.join(scene_path, "part.png")

    img_paths = depths_path + parts_path + [seg_path, rgb_path]

    with open(yaml_path, "r") as fp:
        gt_dict = yaml.load(fp, Loader=yaml.FullLoader)

    for obj_id, obj_prop in tqdm(gt_dict.items(), leave=False,
                                 position=1, desc="  Objects"):
        bbox = obj_prop['bbox']

        for img_path in tqdm(img_paths, leave=False,
                             position=2, desc="    Crops"):
            img_cropped = crop_bbox(img_path, bbox, margin)
            rel_path = os.path.relpath(img_path,
                                       start=scene_path)
            scene_name = os.path.basename(os.path.dirname(scene_path))
            out_path = os.path.join(output_path,
                                    scene_name,
                                    "object_{:03d}".format(obj_id),
                                    rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            img_cropped.save(out_path)


def process_folder(data_path, output_path, margin):

    scene_paths = glob(os.path.join(data_path, "*/"))
    for scene_path in tqdm(scene_paths, position=0, desc="Scenes"):
        process_scene(scene_path, output_path, margin)


if __name__ == "__main__":
    args = parse_args()
    process_folder(args.dataset_dir,
                   args.output_dir,
                   args.bboxes_margin)
