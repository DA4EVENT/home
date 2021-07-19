#!/usr/bin/env python
import os
import pymongo
from datetime import datetime

import tensorflow as tf

from config import configs
from data_loader import get_loader
from data_loader_w_events import get_loader as get_loader_events
from EVFlowNet import EVFlowNet

from termcolor import colored
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds


def dump_to_yaml(args, path):
    with open(path, 'w') as fp:
        for k, v in vars(args).items():
            if isinstance(v, list):
                v = "["+"".join(["{}, ".format(z) for z in v])[:-len(", ")]+"]"
            if isinstance(v, str) and len(v) == 0:
                continue
            if v is None:
                continue
            fp.write("{}: {}\n".format(k, v))


def mongo_compatible(obj):
    if isinstance(obj, dict):
        res = dict()
        for key, value in obj.items():
            key = key.replace(".", ',').replace("$", "S")
            res[key] = mongo_compatible(value)
        return res
    elif isinstance(obj, (list, tuple)):
        return list([mongo_compatible(value) for value in obj])
    return obj


def main():
    args = configs()

    args.restore_path = None
    if args.training_instance:
        if ".ckpt" in args.training_instance:
            training_dir, _ = os.path.splitext(args.training_instance)
            args.restore_path = args.training_instance
        else:
            args.restore_path = tf.train.latest_checkpoint(args.training_instance)
            training_dir = args.training_instance
        print("Restoring checkpoint:", args.restore_path)

        args.load_path = os.path.join(args.load_path, training_dir)
        args.summary_path = os.path.join(args.summary_path, training_dir)
    else:
        args.load_path = os.path.join(args.load_path,
                                      "evflownet_{}_{}".format(datetime.now()
                                                            .strftime("%m%d_%H%M%S"),
                                                               args.exp_name))
        args.summary_path = os.path.join(args.summary_path,
                                         "evflownet_{}_{}".format(datetime.now()
                                                               .strftime("%m%d_%H%M%S"),
                                                               args.exp_name))

        os.makedirs(args.load_path)
        dump_to_yaml(args, os.path.join(args.load_path, "args.yaml"))

    if args.sacred:
        sacred_exp = Experiment(args.exp_name)
        sacred_exp.captured_out_filter = apply_backspaces_and_linefeeds
        conf = vars(args)
        conf.update({'log_dir': args.load_path})
        conf.update({'summary_path': args.summary_path})
        sacred_exp.add_config(mongo_compatible(conf))

        if not args.mongodb_disable:
            url = "{0.mongodb_url}:{0.mongodb_port}".format(args)
            db_name = args.mongodb_name

            overwrite = None
            if args.restore_path is not None:
                client = pymongo.MongoClient(url)
                database = client[db_name]
                runs = database["runs"]
                matches = runs.find({"config.log_dir": args.load_path})
                if matches.count() > 1:
                    raise ValueError("Multiple MongoDB entries found with the specified path!")
                elif matches.count() == 0:
                    raise ValueError("No MongoDB entriy found with the specified path!")
                else:
                    overwrite = matches[0]['_id']

            print(colored('Connect to MongoDB@{}:{}'.format(url, db_name), "green"))
            sacred_exp.observers.append(MongoObserver.create(url=url,
                                                             db_name=db_name,
                                                             overwrite=overwrite))

    if not os.path.exists(args.load_path):
        os.makedirs(args.load_path)
    if not os.path.exists(args.summary_path):
        os.makedirs(args.summary_path)

    # Fix the random seed for reproducibility.
    # Remove this if you are using this code for something else!
    tf.set_random_seed(12345)

    if args.do_aug_rewind:
        if args.no_aug_rot is False:
            raise ValueError("no_aug_rot = False Not supported when do_aug_rewind = True")

        print("Using Event Loader for rewind augmentation!")
        loader_vals = get_loader_events(
            args.data_path, args.batch_size, args.image_width, args.image_height,
            split='train',
            shuffle=True,
            sequence=args.sequences,
            rotation=not args.no_aug_rot,
            rewind=args.do_aug_rewind,
            flip_updown=args.do_aug_flip_updown,
            nskips=args.loader_n_skips,
            binarize_polarity=args.loader_binarize_polarity)
        (events_loader, lengths_loader,
         event_img_loader, prev_img_loader,
         next_img_loader, _, rot_angle, crop_bbox, n_ima) = loader_vals
    else:
        event_img_loader, prev_img_loader, next_img_loader, _, n_ima = get_loader(
            args.data_path, args.batch_size, args.image_width, args.image_height,
            split='train',
            shuffle=True,
            sequence=args.sequences,
            rotation=not args.no_aug_rot,
            flip_updown=args.do_aug_flip_updown,
            nskips=args.loader_n_skips,
            gzip=args.gzip)
    print("Number of images: {}".format(n_ima))
    
    trainer = EVFlowNet(args,
                        event_img_loader,
                        prev_img_loader,
                        next_img_loader,
                        n_ima,
                        is_training=True)

    if args.sacred:
        @sacred_exp.main
        def train_wrapped():
            return trainer.train()
        sacred_exp.run()
    else:
        trainer.train()
    
if __name__ == "__main__":
    main()
