#!/usr/bin/env python3
"""
    Import packages
"""

import argparse
import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from net import ResBase, ResClassifierMMD

from evrepr_head import EvReprHead
from utils import OptimizerManager, EvaluationManager, IteratorWrapper, \
    weights_init, default_param, map_to_device, \
    set_channel, collate_pad_events
#from  utils import  *
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

hp_list = ["MMD_", "Source"+args.source, "Target"+args.target, args.dataset, "GB"+str(args.GB_class), "LR" + str(args.lr),"BS"+str(args.batch_size), "AVG"+str(args.AvgChannels), args.modality]
hp_list = [str(hp) for hp in hp_list]
hp_string = '_'.join(hp_list) + args.suffix
print("Run: " + hp_string)
# Tensorboard summary
args.experiment = args.experiment + "/"+args.task
writer = SummaryWriter(log_dir=os.path.join(args.experiment, hp_string), flush_secs=5)


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
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#train_transform = MyTransformer([int((256 - 224) / 2), int((256 - 224) / 2)], False)
if args.dataset == "ncaltech101":
    print("Normalize with 0 and 1" )
    normalize = Normalize(mean=[0.] * args.channels_event,
                          std=[1.] * args.channels_event)

#train_transform = Compose([Scale(256), RandomHorizontalFlip(), RandomCrop(224),ToTensor(), normalize])
train_transform = Compose([Scale_ReplicateBorder(256), RandomHorizontalFlip(), RandomCrop(224), ToTensor()])
train_transform_with_Rot = Compose([Scale_ReplicateBorder(256), RandomHorizontalFlip(), RandomCrop(224), Rotation(), ToTensor()])

#test_transform = Compose([Scale(256),CenterCrop(224),ToTensor(),normalize])
test_transform = Compose([Scale_ReplicateBorder(256),CenterCrop(224),ToTensor()])

th_train_transform = get_torch_transforms(train_transform)
th_train_transform_with_Rot = get_torch_transforms(train_transform_with_Rot)
th_test_transform = get_torch_transforms(test_transform)

"""
    Prepare datasets
"""


if args.dataset in ["caltech101","ncaltech101"]:
    from data_loader import Caltech101 as loaders
elif args.dataset == "Cifar10":
    from data_loader import CIFAR10 as loaders

# If evrepr="voxelgrid" we expect the modality to be "voxelgrid_#chans"
# assert args.evrepr is None or args.evrepr in args.modality

# Source: training set
train_set_source = loaders(args.data_root_source, path_txt=args.train_file_source, isSource=True, train=True, do_rot=False,
                           transform=train_transform, args=args)
# Target: training set (for entropy)
train_set_target = loaders(args.data_root_target, path_txt=args.train_file_target, isSource=False, train=True,
                           do_rot=False, transform=train_transform, args=args)
# Target: val set
val_set_target = loaders(args.data_root_source, path_txt=args.val_file_target, isSource=False, train=False, do_rot=False,
                         transform=test_transform, args=args)
# Target: test set
test_set_target = loaders(args.data_root_source, path_txt=args.test_file_target, isSource=False, train=False, do_rot=False,
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
# Target: test set (for relative rotation)
rot_set_target = loaders(args.data_root_target, path_txt=args.val_file_target,  isSource=False, train=False,
                                            do_rot=True, transform=train_transform_with_Rot, args=args)

"""
    Prepare data loaders   
"""
#args.batch_size = 16 # da togliere by Mirco
# Source training recognition
train_loader_source = DataLoader(train_set_source,
                                 shuffle=True,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_pad_events,drop_last=True)
# Target train
if args.weight_mmd > 0:
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
                               collate_fn=collate_pad_events)



# Target test recognition
test_loader_target = DataLoader(test_set_target,
                                shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=collate_pad_events)

"""
    Set up network & optimizer
"""
input_dim_F = 2048 if args.net == 'resnet50' else 512

eventHead = EvReprHead(args)
netG = ResBase(architecture=args.net, channels_event=args.channels_event, AvgChannels=args.AvgChannels, device = device)
netF = ResClassifierMMD(input_dim=input_dim_F, class_num=args.class_num, dropout_p=args.dropout_p, extract=False)
netF.apply(weights_init)

net_list = [netG, netF, eventHead]


#Loss and Opt
ce_loss = nn.CrossEntropyLoss()
opt_g = optim.SGD(netG.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
opt_f = optim.SGD(netF.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)
opt_ev = (optim.SGD(eventHead.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
          if eventHead.trainable
          else None)

if args.Adam:
    opt_g = optim.Adam(netG.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_f = optim.Adam(netF.parameters(), lr=args.lr * args.lr_mult, weight_decay=args.weight_decay)
    opt_ev = (optim.Adam(eventHead.parameters(), lr=args.lr, weight_decay=args.weight_decay)
              if eventHead.trainable
              else None)

optims_list = [opt_g, opt_f]
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

print("Ho Forzato il BS ad essere 16, perche per MMD dobbimo concatenare le feat e probabilmento 32 non ci stanno")

# Zero Grad
for op in optims_list:
    op.zero_grad()

for epoch in range(1, args.epoch + 1):
    print("Epoch {} / {}".format(epoch, args.epoch))
    # ========================= TRAINING =========================



    # Train source (recognition)
    train_loader_source_rec_iter = train_loader_source
    # Train target (entropy)
    if args.weight_mmd > 0:
        train_target_loader_iter = IteratorWrapper(train_loader_target)
    # Val target
    val_target_loader_iter = IteratorWrapper(val_loader_target)


    with tqdm(total=len(train_loader_source), desc="Train") as pb:
        for batch_num, (img, img_label_source) in enumerate(train_loader_source_rec_iter):
            #if img.size(0) != args.batch_size:
            #   break

            # The optimization step is performed by OptimizerManager
            with OptimizerManager(optims_list, batch_num, args.num_accumulation):
                # Compute source features

                img, img_label_source = map_to_device(device, (img, img_label_source))
                img = eventHead(img, transform=th_train_transform)
                feat, _ = netG(img)
                features_source = feat
                del img

                # Entropy loss

                # Compute target features
                img, lab = train_target_loader_iter.get_next()
                img, lab = map_to_device(device, (img, lab))
                del lab
                img = eventHead(img, transform=th_train_transform)
                feat_target, _ = netG(img)
                features_target = feat_target
                del img

                #Compute Classification and MMD losses
                logits, mmd_loss = netF(features_source, features_target)
                #del features_source, features_target
                loss_rec = ce_loss(logits, img_label_source)
                loss = (loss_rec + mmd_loss.mean() * args.weight_mmd) * SCALE_Accumulation
                loss.backward()

                pb.update(1)

    # ========================= VALIDATION =========================

    # Recognition - target
    #actual_test_batches = min(len(test_loader_target), args.test_batches)
    with EvaluationManager(net_list), tqdm(total=len(val_loader_target), desc="Val") as pb:
        val_target_loader_iter = iter(val_loader_target)
        correct = 0.0
        num_predictions = 0.0
        val_loss = 0.0
        for num_batch, (img, img_label_target) in enumerate(val_target_loader_iter):
            # By default validate only on 100 batches
            #if num_batch >= args.test_batches:
            #    break

            # Compute features
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



    # Save the best model
    if (Best_Acc_Val < val_acc) and args.SaveModel:
        Best_Acc_Val = val_acc
        Best_Epoch = epoch
        if epoch % 1 == 0:
            if not os.path.exists(args.snapshot):
                os.mkdir(args.snapshot)

            if eventHead.trainable:
                torch.save(eventHead.state_dict(), os.path.join(
                    args.snapshot,
                    hp_string + "_eventHead_" + "_epoch" + str(epoch) + ".pth"))

            torch.save(netG.state_dict(),
                       os.path.join(args.snapshot, hp_string + "_netG_" + "_epoch" + str(epoch) + ".pth"))

            torch.save(netF.state_dict(), os.path.join(args.snapshot, hp_string + "_netF_"  + "_epoch" + str(epoch) + ".pth"))

print("The Best Acc: ", Best_Acc_Val)
print("Epoch: ", Best_Epoch)


print("Starting Test on target...")
# Val target
test_target_loader_iter = IteratorWrapper(test_loader_target)
with EvaluationManager(net_list), tqdm(total=len(test_loader_target), desc="TestRecS") as pb:
    test_target_loader_iter = iter(test_loader_target)
    correct = 0.0
    num_predictions = 0.0
    test_loss = 0.0
    for num_batch, (img, img_label_target) in enumerate(test_target_loader_iter):
        # By default validate only on 100 batches
        # if num_batch >= args.test_batches:
        #    break

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
    print("Test Target accuracy (recognition): {}".format(test_acc))
    with open(os.path.join(os.path.join(args.experiment, hp_string), 'val_precision.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc_Test: %.4f%%\n" %
                (epoch,args.epoch,test_acc))