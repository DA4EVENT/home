#!/usr/bin/env python3
"""
    Import packages
"""

import numpy as np
import argparse
import os


import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from net import ResBase, ResClassifier

from evrepr_head import EvReprHead
from utils import OptimizerManager, EvaluationManager, IteratorWrapper, \
    weights_init, default_param, map_to_device, \
    entropy_loss, set_channel, collate_pad_events

from args import add_base_args
from spatialTransforms_torch import get_torch_transforms
import torch.nn.parallel
from tqdm import tqdm



# Parse arguments
parser = argparse.ArgumentParser()
add_base_args(parser)
parser.add_argument('--test_batches', default=100, type=int)
args = parser.parse_args()
args.channels_event = set_channel(args, args.modality)

##GPU
if args.gpu is not None:
    print('Using only these GPUs: {}'.format(args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(os.environ['CUDA_VISIBLE_DEVICES'])

# Load default paths if needed
default_param(args)
SCALE_Accumulation = 1 / args.num_accumulation

#################################
#                               #
#        Name Experiment        #
#                               #
#################################

hp_list = ["Single_Modal", "Source_"+args.source, "Target_"+args.target, "LR" + str(args.lr),"BS"+str(args.batch_size), "AVG"+str(args.AvgChannels), args.modality]
hp_list = [str(hp) for hp in hp_list]
hp_string = '_'.join(hp_list) + args.suffix
print("Run: " + hp_string)
# Tensorboard summary
args.experiment = args.experiment + "/"+args.task
writer = SummaryWriter(log_dir=os.path.join(args.experiment, hp_string), flush_secs=5)
args.snapshot = os.path.join(args.experiment, hp_string, "snapshot")


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE", device)
#################################
#                               #
#       Pre-Processing          #
#                               #
#################################

from spatialTransforms import (Compose, ToTensor, CenterCrop, Normalize,
                               RandomHorizontalFlip, RandomCrop, Rotation, Scale_ReplicateBorder)
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], imagenet_norm = args.imagenet_norm)

train_transform = Compose([Scale_ReplicateBorder(256), RandomHorizontalFlip(), RandomCrop(224), ToTensor(),normalize])
train_transform_with_Rot = Compose([Scale_ReplicateBorder(256), RandomHorizontalFlip(), RandomCrop(224), Rotation(), ToTensor(),normalize])
test_transform = Compose([Scale_ReplicateBorder(256),CenterCrop(224),ToTensor(),normalize])



th_train_transform = get_torch_transforms(train_transform, args)
th_train_transform_with_Rot = get_torch_transforms(train_transform_with_Rot, args)
th_test_transform = get_torch_transforms(test_transform, args)

"""
    Prepare datasets
"""
from data_loader import ROD as loaders


# Source: training set
train_set_source = loaders(args.data_root_source, path_txt=args.train_file_source, isSource=True, train=True, do_rot=False,
                           transform=train_transform, args=args)
# Target: training set (for entropy)
train_set_target = loaders(args.data_root_target, path_txt=args.train_file_target, isSource=False, train=True,
                           do_rot=False, transform=train_transform, args=args)
# Target: val set
val_set_target = loaders(args.data_root_source, path_txt=args.val_file_target, isSource=True, train=False, do_rot=False,
                         transform=test_transform, args=args)
# Target: test set
test_set_target = loaders(args.data_root_target, path_txt=args.test_file_target, isSource=False, train=False, do_rot=False,
                          transform=test_transform, args=args)


# Source: training set (for relative rotation)
rot_set_source = loaders(args.data_root_source, path_txt=args.train_file_source, isSource=True, train=True, do_rot=True,
                                             transform=train_transform_with_Rot, args=args)
# Source: test set (for relative rotation)
rot_test_set_source = loaders(args.data_root_source, path_txt=args.val_file_source,  isSource=True, train=False, do_rot=True,
                                                  transform=train_transform_with_Rot, args=args)
# Target: training  (for relative rotation)
rot_set_target_train = loaders(args.data_root_target, path_txt=args.train_file_target,  isSource=False, train=True,
                                            do_rot=True, transform=train_transform_with_Rot, args=args)
# Target: test set (for relative rotation) (test on test Real ROD)
rot_set_target = loaders(args.data_root_target, path_txt=args.test_file_target,  isSource=False, train=False,
                                            do_rot=True, transform=train_transform_with_Rot, args=args)

"""
    Prepare data loaders   
"""

# Source training recognition
train_loader_source = DataLoader(train_set_source,
                                 shuffle=True,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_pad_events,drop_last=True)
# Target train
if args.weight_ent > 0:
    train_loader_target = DataLoader(train_set_target,
                                     shuffle=True,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     collate_fn=collate_pad_events,drop_last=True)


# Target Val recognition
val_loader_target = DataLoader(val_set_target,
                               shuffle=False,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               collate_fn=collate_pad_events, drop_last=True)



# Target test recognition
test_loader_target = DataLoader(test_set_target,
                                shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=collate_pad_events, drop_last=False)

# Source rot
if args.weight_rot > 0:

    rot_source_loader = DataLoader(rot_set_source,
                                   shuffle=True,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_pad_events,drop_last=True)

    rot_val_source_loader = DataLoader(rot_test_set_source,
                                        shuffle=True,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        collate_fn=collate_pad_events,drop_last=True)

    # Target rot

    rot_target_loader = DataLoader(rot_set_target_train,
                                   shuffle=True,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_pad_events,drop_last=True)

    rot_val_target_loader = DataLoader(rot_set_target,
                                        shuffle=True,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        collate_fn=collate_pad_events,drop_last=True)

"""
    Set up network & optimizer
"""
input_dim_F = 2048 if args.net == 'resnet50' else 512

eventHead = EvReprHead(args)
netG = ResBase(architecture=args.net, channels_event=args.channels_event, AvgChannels=args.AvgChannels, device = device)
netF = ResClassifier(input_dim=input_dim_F, class_num=args.class_num, dropout_p=args.dropout_p, extract=False)
netF_rot = ResClassifier(input_dim=input_dim_F, class_num=4, dropout_p=args.dropout_p, extract=False)
netF_rot.apply(weights_init)
netF.apply(weights_init)

net_list = [netG, netF, eventHead]
if args.weight_rot > 0.0:
    net_list.append(netF_rot)

#Loss and Opt
ce_loss = nn.CrossEntropyLoss()
opt_g = optim.SGD(netG.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
opt_f = optim.SGD(netF.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)
opt_f_rot = optim.SGD(netF_rot.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)
opt_ev = (optim.SGD(eventHead.parameters(), lr=args.lr * args.est_lr_mul, momentum=0.9, weight_decay=args.weight_decay)
          if eventHead.trainable
          else None)

if args.Adam:
    opt_g = optim.Adam(netG.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_f = optim.Adam(netF.parameters(), lr=args.lr * args.lr_mult, weight_decay=args.weight_decay)
    opt_f_rot = optim.Adam(netF_rot.parameters(), lr=args.lr * args.lr_mult, weight_decay=args.weight_decay)
    opt_ev = (optim.Adam(eventHead.parameters(), lr=args.lr, weight_decay=args.weight_decay)
              if eventHead.trainable
              else None)

optims_list = [opt_g, opt_f, opt_f_rot]
if opt_ev is not None:
    optims_list += [opt_ev]


######################
#                    #
#   Data Parallel    #
#                    #
######################

for i, model in enumerate(net_list):
    print("DataParallel")
    net_list[i] = torch.nn.DataParallel(net_list[i]).to(device)

netG, netF, eventHead = net_list[:3]
if args.weight_rot > 0.0:
    netF_rot = net_list[-1]

######################
#                    #
#    Save All Param  #
#                    #
######################

with open(os.path.join(os.path.join(args.experiment, hp_string), './parameters.txt'), 'w') as myfile:
    for arg in sorted(vars(args)):
        myfile.write('{:>20} = {}\n'.format(arg, getattr(args, arg)))
        print('{:>20} = {}'.format(arg, getattr(args, arg)))

######################                  
#                    #
#    Training loop   #
#                    #
######################

Best_Acc_Val = 0
Best_Epoch = 0
if args.num_accumulation != 1:
    print("Accumulation n  --> ", args.num_accumulation)

# Zero Grad
for op in optims_list:
    op.zero_grad()

for epoch in range(1, args.epoch + 1):
    print("Epoch {} / {}".format(epoch, args.epoch))
    # ========================= TRAINING =========================
    # Train source (recognition)
    train_loader_source_rec_iter = train_loader_source
    # Train target (entropy)
    if args.weight_ent > 0:
        train_target_loader_iter = IteratorWrapper(train_loader_target)
    # Val target
    val_target_loader_iter = IteratorWrapper(val_loader_target)

    # (rotation)
    if args.weight_rot > 0:
        # Source (rotation)
        rot_source_loader_iter = IteratorWrapper(rot_source_loader)
        # Target (rotation)
        rot_target_loader_iter = IteratorWrapper(rot_target_loader)

    with tqdm(total=len(train_loader_source), desc="Train") as pb:
        for batch_num, (img, img_label_source) in enumerate(train_loader_source_rec_iter):

            # The optimization step is performed by OptimizerManager
            with OptimizerManager(optims_list, batch_num, args.num_accumulation):
                # Compute source features

                img, img_label_source = map_to_device(device, (img, img_label_source))
                img = eventHead(img, transform=th_train_transform)
                feat, _ = netG(img)
                features_source = feat
                logits = netF(features_source)

                # Classification loss
                loss_rec = ce_loss(logits, img_label_source) * SCALE_Accumulation
                loss_rec.backward()
                del img, img_label_source, feat, logits

                # Entropy loss
                if args.weight_ent > 0.:
                    # Compute target features
                    img, lab = train_target_loader_iter.get_next()
                    img, lab = map_to_device(device, (img, lab))
                    del lab
                    img = eventHead(img, transform=th_train_transform)
                    feat, _ = netG(img)
                    features_target = feat
                    logits = netF(features_target)

                    loss_ent = entropy_loss(logits) * SCALE_Accumulation
                    del img, features_target, feat, logits
                    # Backpropagate
                    loss = args.weight_ent * loss_ent
                    loss.backward()

                else:
                    loss_ent = 0

                # Relative Rotation
                if args.weight_rot > 0.0:
                    # Load batch: rotation, source
                    img, _, rot_label = rot_source_loader_iter.get_next()
                    img, rot_label = map_to_device(device, (img, rot_label))

                    # Compute features (with pooling!)
                    img = eventHead(img, transform=th_train_transform_with_Rot,
                                    rot=rot_label)
                    feat, _ = netG(img)
                    # Prediction
                    logits_rot = netF_rot(feat)

                    # Classification loss for the rleative rotation task
                    loss_rot = ce_loss(logits_rot, rot_label) * SCALE_Accumulation
                    loss = args.weight_rot * loss_rot
                    # Backpropagate
                    loss.backward()

                    loss_rot = loss_rot.item()

                    del img, rot_label, feat, logits_rot, loss

                    # Load batch: rotation, target
                    img, _, rot_label = rot_target_loader_iter.get_next()
                    img, rot_label = map_to_device(device, (img, rot_label))

                    # Compute features (with pooling!)
                    img = eventHead(img, transform=th_train_transform_with_Rot,
                                    rot=rot_label)
                    feat, _ = netG(img)
                    # Prediction
                    logits_rot = netF_rot(feat)

                    # Classification loss for the rleative rotation task
                    loss = args.weight_rot * ce_loss(logits_rot, rot_label) * SCALE_Accumulation
                    # Backpropagate
                    loss.backward()

                    del img, rot_label, feat, logits_rot, loss

                pb.update(1)

    # ========================= VALIDATION =========================
    if epoch % 5 == 0:

        # Recognition - target
        with EvaluationManager(net_list), tqdm(total=len(val_loader_target), desc="Val") as pb:
            val_target_loader_iter = iter(val_loader_target)
            correct = 0.0
            num_predictions = 0.0
            val_loss = 0.0
            for num_batch, (img, img_label_target) in enumerate(val_target_loader_iter):
                # By default validate only on 100 batches
                #if num_batch >= args.test_batches:
                #    break

                # Compute source features
                img, img_label_target = map_to_device(device, (img, img_label_target))
                img = eventHead(img, transform=th_test_transform)
                feat, _ = netG(img)
                features_target = feat

                # Compute predictions
                preds = netF(features_target)

                val_loss += ce_loss(preds, img_label_target).item()
                correct += (torch.argmax(preds, dim=1) == img_label_target).sum().item()
                num_predictions += preds.shape[0]

                pb.update(1)

            val_acc = correct / num_predictions
            val_loss = val_loss / num_predictions


            print("Epoch: {} - Validation target accuracy (recognition): {}".format(epoch, val_acc))

        del img, img_label_target, feat, preds

        writer.add_scalar("Loss/train", loss_rec.item(), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        with open(os.path.join(os.path.join(args.experiment, hp_string), 'val_precision.txt'), 'a+') as f:
            f.write("[%d/%d]\tAccVal: %.4f%%\n" %
                    (epoch,args.epoch,val_acc))


        # Relative Rotation
        if args.weight_rot > 0.0:

            # Rotation - source
            cf_matrix = np.zeros([4, 4])
            #actual_test_batches = min(len(rot_val_source_loader), args.test_batches)
            with EvaluationManager(net_list), tqdm(total=len(rot_val_source_loader), desc="Val Rot S") as pb:
                rot_val_source_loader_iter = iter(rot_val_source_loader)
                correct = 0.0
                num_predictions = 0.0
                for num_val_batch, (img, _, rot_label) in enumerate(rot_val_source_loader_iter):
                    #if num_val_batch > args.test_batches:
                    #    break
                    img, rot_label = map_to_device(device, (img, rot_label))

                    # Compute features (with pooling)
                    img = eventHead(img, transform=th_train_transform_with_Rot,
                                    rot=rot_label)
                    feat, _ = netG(img)
                    # Compute predictions
                    preds = netF_rot(feat)

                    val_loss_rot = ce_loss(preds, rot_label).item()
                    correct += (torch.argmax(preds, dim=1) == rot_label).sum().item()
                    num_predictions += preds.shape[0]

                    pb.update(1)

                del img, rot_label, preds, feat

                rot_val_acc = correct / num_predictions
                print("Epoch: {} - Validation source rotation accuracy: {}".format(epoch, rot_val_acc))

            # Rotation - target
            #actual_test_batches = min(len(rot_val_target_loader), args.test_batches)
            with EvaluationManager(net_list), tqdm(total=len(rot_val_target_loader), desc="Val Rot T") as pb:
                rot_val_target_loader_iter = iter(rot_val_target_loader)
                correct = 0.0
                val_loss_rot = 0.0
                num_predictions = 0.0
                for num_val_batch, (img, _, rot_label) in enumerate(rot_val_target_loader_iter):
                    #if num_val_batch > args.test_batches:
                    #    break

                    img, rot_label = map_to_device(device, (img, rot_label))

                    # Compute features (with pooling)
                    img = eventHead(img, transform=th_train_transform_with_Rot,
                                    rot=rot_label)
                    feat, _ = netG(img)
                    # Compute predictions
                    preds = netF_rot(feat)

                    val_loss_rot += ce_loss(preds, rot_label).item()
                    correct += (torch.argmax(preds, dim=1) == rot_label).sum().item()
                    num_predictions += preds.shape[0]

                    pb.update(1)

                rot_val_acc = correct / num_predictions
                val_loss_rot = val_loss_rot / num_predictions
                print("Epoch: {} - Validation target rotation accuracy: {}".format(epoch, rot_val_acc))

            del img, rot_label, preds

            writer.add_scalar("Loss/rot", loss_rot, epoch)
            writer.add_scalar("Loss/rot_val", val_loss_rot, epoch)
            writer.add_scalar("Accuracy/rot_val", rot_val_acc, epoch)

    # Save the best model
    if epoch % 5 == 0:
    #SAVE THE MODEL
        if not os.path.exists(args.snapshot):
            os.mkdir(args.snapshot)
        if eventHead.module.trainable:
            torch.save(eventHead.state_dict(), os.path.join(
                args.snapshot,
                hp_string + "_eventHead_epoch" + str(epoch) + ".pth"))

        torch.save(netG.state_dict(),
                   os.path.join(args.snapshot, hp_string + "_netG_" + args.modality + "_epoch" + str(epoch) + ".pth"))

        torch.save(netF.state_dict(), os.path.join(args.snapshot, hp_string + "_netF_epoch" + str(epoch) + ".pth"))
        if args.weight_rot > 0.0:
            torch.save(netF_rot.state_dict(),
                   os.path.join(args.snapshot, hp_string + "_netF_rot_epoch" + str(epoch) + ".pth"))



print("Starting Test on target...")
with open(os.path.join(os.path.join(args.experiment, hp_string), 'val_precision.txt'), 'a+') as f:
    f.write("")
    f.write("Starting Test on target...")
    f.write("")

####################
#  Load The Models #
####################
for epoch in range(5, args.epoch + 1, 5):
    # Load network weights
    netG.load_state_dict(
        torch.load(os.path.join(args.snapshot, hp_string + "_netG_" + args.modality + "_epoch" + str(epoch) + ".pth"),
                   map_location=device))

    netF.load_state_dict(
        torch.load(os.path.join(args.snapshot, hp_string + "_netF_epoch" + str(epoch) + ".pth"),
                   map_location=device))

    if eventHead.module.trainable:
        eventHead.load_state_dict(
            torch.load(os.path.join(args.snapshot, hp_string + "_eventHead_epoch" + str(epoch) + ".pth"),
                       map_location=device))




    # Val target
    test_target_loader_iter = IteratorWrapper(test_loader_target)
    with EvaluationManager(net_list), tqdm(total=len(test_loader_target), desc="TestRecS") as pb:
        test_target_loader_iter = iter(test_loader_target)
        correct = 0.0
        num_predictions = 0.0
        test_loss = 0.0
        for num_batch, (img, img_label_target) in enumerate(test_target_loader_iter):

            # Compute source features
            img, img_label_target = map_to_device(device, (img, img_label_target))
            img = eventHead(img, transform=th_test_transform)
            feat, _ = netG(img)
            features_target = feat

            # Compute predictions
            preds = netF(features_target)

            test_loss += ce_loss(preds, img_label_target).item()
            correct += (torch.argmax(preds, dim=1) == img_label_target).sum().item()
            num_predictions += preds.shape[0]

            pb.update(1)

        test_acc = correct / num_predictions
        test_loss = test_loss / num_predictions
        print("Epoch: {} - Test Target accuracy (recognition): {}".format(epoch, test_acc))
        with open(os.path.join(os.path.join(args.experiment, hp_string), 'val_precision.txt'), 'a+') as f:
            f.write("[%d/%d]\tAcc_Test: %.5f%%\n" %
                    (epoch,args.epoch,test_acc))