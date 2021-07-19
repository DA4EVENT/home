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

from net import ResBase, ResClassifier, GradientReversalLayer

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

hp_list = ["GRL_Multi-Modal", "Source_" + args.source, "Target_" + args.target, "LR" + str(args.lr),"BS"+str(args.batch_size), "AVG"+str(args.AvgChannels), args.modality]
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


from spatialTransforms import (Compose, ToTensor, CenterCrop, Scale_ReplicateBorder, Normalize,
                               RandomHorizontalFlip, RandomCrop)
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], imagenet_norm = args.imagenet_norm)

train_transform = Compose([Scale_ReplicateBorder(256), RandomHorizontalFlip(), RandomCrop(224), ToTensor(),normalize])
test_transform = Compose([Scale_ReplicateBorder(256),CenterCrop(224),ToTensor(),normalize])

th_train_transform = get_torch_transforms(train_transform,args)
th_test_transform = get_torch_transforms(test_transform,args)

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


"""
    Prepare data loaders
    Source Recognition
    Source Discriminator (1/2)
    Target Entropy
    Target Discriminator (1/2)
    
    Target Val
    Target Test
"""


# Source training recognition
train_loader_source = DataLoader(train_set_source,
                                 shuffle=True,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_pad_events,drop_last=True)

# Source training discriminator (1/2)
if args.weight_grl > 0:
    train_loader_source_disc = DataLoader(train_set_source,
                                 shuffle=True,
                                 batch_size=args.batch_size // 2,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_pad_events, drop_last=True)

# Target train
if args.weight_ent > 0:
    train_loader_target = DataLoader(train_set_target,
                                     shuffle=True,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     collate_fn=collate_pad_events,drop_last=True)
# Target training discriminator (1/2)
if args.weight_grl > 0:
    train_loader_target_disc = DataLoader(train_set_target,
                                 shuffle=True,
                                 batch_size=args.batch_size // 2,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_pad_events, drop_last=True)


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


"""
    Set up network & optimizer
"""

args.channels_event = set_channel(args, args.modality)
input_dim_F = 2048 if args.net == 'resnet50' else 512

eventHead = EvReprHead(args)

netG_rgb = ResBase(architecture=args.net, device = device)
netG_event = ResBase(architecture=args.net, channels_event=args.channels_event, AvgChannels=args.AvgChannels, device = device)

netF = ResClassifier(input_dim=input_dim_F * 2, class_num=args.class_num, dropout_p=args.dropout_p, extract=False)
netF.apply(weights_init)

# ADD GRL
netF_disc = ResClassifier(input_dim=input_dim_F*2, class_num=2, dropout_p=args.dropout_p, extract=False)
netGRL = GradientReversalLayer()
netF_disc.apply(weights_init)


net_list = [netG_rgb, netG_event, netF, eventHead]
if args.weight_grl > 0.0:
    net_list.append(netF_disc)


#Loss and Opt
ce_loss = nn.CrossEntropyLoss()
opt_g_rgb = optim.SGD(netG_rgb.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
opt_g_event = optim.SGD(netG_event.parameters(), lr=args.lr * args.lr_event_mult, momentum=0.9, weight_decay=args.weight_decay)
opt_f = optim.SGD(netF.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)
opt_f_disc = optim.SGD(netF_disc.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)
opt_ev = (optim.SGD(eventHead.parameters(), lr=args.lr * args.est_lr_mul, momentum=0.9, weight_decay=args.weight_decay)
          if eventHead.trainable
          else None)


optims_list = [opt_g_rgb, opt_g_event, opt_f, opt_f_disc]
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

netG_rgb, netG_event, netF, eventHead = net_list[:4]
if args.weight_grl > 0.0:
    netF_disc = net_list[-1]

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

if args.num_accumulation != 1:
    print("Accumulation n  --> ", args.num_accumulation)

# For lambda computation
iteration_counter = 0
tot_iterations = args.epoch * len(train_loader_source)

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
    # Train Source/Target (GRL)
    if args.weight_grl > 0:
        train_source_loader_disc_iter = IteratorWrapper(train_loader_source_disc)
        train_target_loader_disc_iter = IteratorWrapper(train_loader_target_disc)
    # Test target
    test_target_loader_iter = IteratorWrapper(test_loader_target)


    with tqdm(total=len(train_loader_source), desc="Train") as pb:
        for batch_num, (img_rgb, img_event, img_label_source) in enumerate(train_loader_source_rec_iter):


            # The optimization step is performed by OptimizerManager
            with OptimizerManager(optims_list, batch_num, args.num_accumulation):
                # Compute source features
                img_rgb, img_event, img_label_source = map_to_device(device, (img_rgb, img_event, img_label_source))
                img_event = eventHead(img_event, transform=th_train_transform)

                feat_rgb, _ = netG_rgb(img_rgb)
                feat_event, _ = netG_event(img_event)
                features_source = torch.cat((feat_rgb, feat_event), 1)
                logits = netF(features_source)

                # Classification los
                loss_rec = ce_loss(logits, img_label_source) * SCALE_Accumulation
                loss_rec.backward()
                del img_rgb, img_event, img_label_source, feat_rgb, features_source, feat_event, logits

                # Entropy loss
                if args.weight_ent > 0.:
                    # Compute target features
                    img_rgb, img_event, _ = train_target_loader_iter.get_next()
                    img_rgb, img_event = map_to_device(device, (img_rgb, img_event))
                    img_event = eventHead(img_event, transform=th_train_transform)

                    feat_rgb, _ = netG_rgb(img_rgb)
                    feat_event, _ = netG_event(img_event)
                    features_target = torch.cat((feat_rgb, feat_event), 1)
                    logits = netF(features_target)

                    loss_ent = entropy_loss(logits) * SCALE_Accumulation
                    del img_rgb, img_event, features_target, feat_rgb, feat_event, logits
                    # Backpropagate
                    loss = args.weight_ent * loss_ent
                    loss.backward()
                else:
                    loss_ent = 0



                # GRL
                if args.weight_grl > 0.0:
                    # Load batch: GRL, source
                    img_rgb, img_event, _ = train_source_loader_disc_iter.get_next()
                    img_rgb, img_event = map_to_device(device, (img_rgb, img_event))

                    # Compute features (with pooling!)
                    img_event = eventHead(img_event, transform=th_train_transform)

                    lambda_v = 2 / (1 + np.exp(-10 * iteration_counter / tot_iterations)) - 1
                    iteration_counter += 1

                    feat_rgb, _ = netG_rgb(img_rgb)
                    feat_event, _ = netG_event(img_event)
                    # Prediction
                    logits_disc = netF_disc(netGRL.apply(torch.cat((feat_rgb, feat_event), 1), lambda_v))

                    # Classification loss for the GRL task
                    loss_grl = ce_loss(logits_disc, torch.zeros(logits_disc.size(0), dtype=torch.int64).to(device))

                    loss = args.weight_grl * loss_grl * SCALE_Accumulation

                    # Backpropagate
                    loss.backward()

                    del img_rgb, img_event, feat_rgb, feat_event, logits_disc, loss

                    # Load batch: GRL, target
                    img_rgb, img_event, _  = train_target_loader_disc_iter.get_next()
                    img_rgb, img_event = map_to_device(device, (img_rgb, img_event))

                    img_event = eventHead(img_event, transform=th_train_transform )

                    # Compute features (with pooling!)
                    feat_rgb, _ = netG_rgb(img_rgb)
                    feat_event, _ = netG_event(img_event)
                    # Prediction
                    logits_disc = netF_disc(netGRL.apply(torch.cat((feat_rgb, feat_event), 1), lambda_v))

                    loss_grl = ce_loss(logits_disc, torch.ones(logits_disc.size(0), dtype=torch.int64).to(device))
                    loss = args.weight_grl * loss_grl * SCALE_Accumulation
                    loss.backward()

                del img_rgb, img_event, feat_rgb, feat_event, logits_disc, loss

                pb.update(1)

    # ========================= VALIDATION =========================
    if epoch % 5 == 0:

        # Recognition - source
        with EvaluationManager(net_list), tqdm(total=len(val_loader_target), desc="Val") as pb:
            val_target_loader_iter = iter(val_loader_target)
            correct = 0.0
            num_predictions = 0.0
            val_loss = 0.0
            for num_batch, (img_rgb, img_event, img_label_source) in enumerate(val_target_loader_iter):

                # Compute source features
                img_rgb, img_event, img_label_source = map_to_device(device, (img_rgb, img_event, img_label_source))
                img_event = eventHead(img_event, transform=th_test_transform)

                feat_rgb, _ = netG_rgb(img_rgb)
                feat_event, _ = netG_event(img_event)
                features_source = torch.cat((feat_rgb, feat_event), 1)

                # Compute predictions
                preds = netF(features_source)

                val_loss += ce_loss(preds, img_label_source).item()
                correct += (torch.argmax(preds, dim=1) == img_label_source).sum().item()
                num_predictions += preds.shape[0]

                pb.update(1)

            val_acc = correct / num_predictions
            val_loss = val_loss / args.test_batches
            print("Epoch: {} - Validation source accuracy (recognition): {}".format(epoch, val_acc))

        del img_rgb, img_event, img_label_source, feat_rgb, feat_event, preds

        writer.add_scalar("Loss/train", loss_rec.item(), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if args.weight_grl > 0.0:
            ### Source Test GRL

            cf_matrix = np.zeros([4, 4])
            with EvaluationManager(net_list), tqdm(total=len(val_loader_target), desc="Val GRL S") as pb:
                test_source_loader_iter = iter(val_loader_target)
                correct = 0.0
                num_predictions = 0.0
                for num_val_batch, (img_rgb, img_event, _) in enumerate(test_source_loader_iter):

                    img_rgb, img_event = map_to_device(device, (img_rgb, img_event))
                    img_event = eventHead(img_event, transform=th_test_transform)

                    # Compute features (without pooling)
                    feat_rgb, _ = netG_rgb(img_rgb)
                    feat_event, _ = netG_event(img_event)
                    # Compute predictions
                    preds = netF_disc(torch.cat((feat_rgb, feat_event), 1))
                    domain_label = torch.zeros(img_rgb.size(0), dtype=torch.int64).to(device)

                    val_loss_disc = ce_loss(preds, domain_label).item()
                    correct += (torch.argmax(preds, dim=1) == domain_label).sum().item()
                    num_predictions += preds.shape[0]

                    pb.update(1)

                del img_rgb, img_event, domain_label, preds, feat_event, feat_rgb

                val_acc_disc_source = correct / num_predictions
                source_num_prediction = num_predictions
                val_loss_source_disc = val_loss_disc / num_predictions
                #print("Epoch: {} - Validation source rotation accuracy: {}".format(epoch, rot_val_acc))

            # GRL - target
            with EvaluationManager(net_list), tqdm(total=len(test_loader_target), desc="Val GRL T") as pb:
                test_target_loader_iter = iter(test_loader_target)
                correct = 0.0
                val_loss_rot = 0.0
                num_predictions = 0.0
                for num_val_batch, (img_rgb, img_event, _) in enumerate(test_target_loader_iter):

                    img_rgb, img_event = map_to_device(device, (img_rgb, img_event))
                    img_event = eventHead(img_event, transform=th_test_transform)

                    # Compute features (without pooling)
                    feat_rgb, _ = netG_rgb(img_rgb)
                    feat_event, _ = netG_event(img_event)
                    # Compute predictions
                    preds = netF_disc(torch.cat((feat_rgb, feat_event), 1))
                    domain_label = torch.ones(img_rgb.size(0), dtype=torch.int64).to(device)

                    val_loss_disc += ce_loss(preds, domain_label).item()
                    correct += (torch.argmax(preds, dim=1) == domain_label).sum().item()
                    num_predictions += preds.shape[0]

                    pb.update(1)

                val_acc_disc_target = correct / num_predictions
                target_num_prediction = num_predictions
                val_loss_target_disc = val_loss_disc / num_predictions
                #print("Epoch: {} - Validation target rotation accuracy: {}".format(epoch, rot_val_acc))

            del img_rgb, img_event, domain_label, preds, feat_event, feat_rgb

            val_acc_disc = ((val_acc_disc_target * target_num_prediction) +
                            (val_acc_disc_source * source_num_prediction)) / \
                           (target_num_prediction + source_num_prediction)

            writer.add_scalar("Accuracy/disc_val", val_acc_disc, epoch)

    # Save models
    if epoch % 5 == 0:
        # SAVE THE MODEL
        if not os.path.exists(args.snapshot):
            os.mkdir(args.snapshot)
        if eventHead.module.trainable:
            torch.save(eventHead.state_dict(), os.path.join(
                args.snapshot,
                hp_string + "_eventHead_epoch" + str(epoch) + ".pth"))

        torch.save(netG_rgb.state_dict(),
                   os.path.join(args.snapshot,
                                hp_string + "_netG_" + args.modality.split("-")[0] + "_epoch" + str(epoch) + ".pth"))
        torch.save(netG_event.state_dict(),
                   os.path.join(args.snapshot,
                                hp_string + "_netG_" + args.modality.split("-")[1] + "_epoch" + str(epoch) + ".pth"))
        torch.save(netF.state_dict(), os.path.join(args.snapshot, hp_string + "_netF_epoch" + str(epoch) + ".pth"))
        if args.weight_grl > 0.0:
            torch.save(netF_disc.state_dict(),
                       os.path.join(args.snapshot, hp_string + "_netF_GRL_epoch" + str(epoch) + ".pth"))



print("Starting Test on target...")
with open(os.path.join(os.path.join(args.experiment, hp_string), 'val_precision.txt'), 'a+') as f:
    f.write("")
    f.write("Starting Test on target...")

####################
#  Load The Models #
####################
for epoch in range(5, args.epoch + 1, 5):
    # Load network weights
    netG_rgb.load_state_dict(
        torch.load(os.path.join(args.snapshot, hp_string + "_netG_" +args.modality.split("-")[0] + "_epoch" + str(epoch) + ".pth"),
                   map_location=device))

    netG_event.load_state_dict(
        torch.load(os.path.join(args.snapshot, hp_string + "_netG_" + args.modality.split("-")[1] + "_epoch" + str(epoch) + ".pth"),
                   map_location=device))

    netF.load_state_dict(
        torch.load(os.path.join(args.snapshot, hp_string + "_netF_epoch" + str(epoch) + ".pth"),
                   map_location=device))

    if eventHead.module.trainable:
        eventHead.load_state_dict(
            torch.load(os.path.join(args.snapshot, hp_string + "_eventHead_epoch" + str(epoch) + ".pth"),
                       map_location=device))

    # Test target
    test_target_loader_iter = IteratorWrapper(test_loader_target)
    with EvaluationManager(net_list), tqdm(total=len(test_loader_target), desc="TestRecS") as pb:
        test_target_loader_iter = iter(test_loader_target)
        correct = 0.0
        num_predictions = 0.0
        test_loss = 0.0
        for num_batch, (img_rgb, img_event, img_label_target) in enumerate(test_target_loader_iter):

            # Compute source features
            img_rgb, img_event, img_label_target = map_to_device(device, (img_rgb, img_event, img_label_target))
            img_event = eventHead(img_event, transform=th_test_transform)

            feat_rgb, _ = netG_rgb(img_rgb)
            feat_event, _ = netG_event(img_event)
            features_source = torch.cat((feat_rgb, feat_event), 1)

            # Compute predictions
            preds = netF(features_source)

            test_loss += ce_loss(preds, img_label_target).item()
            correct += (torch.argmax(preds, dim=1) == img_label_target).sum().item()
            num_predictions += preds.shape[0]

            pb.update(1)

        test_acc = correct / num_predictions
        test_loss = test_loss / num_predictions
        print("Test Target accuracy (recognition): {}".format(test_acc))
        with open(os.path.join(os.path.join(args.experiment, hp_string), 'val_precision.txt'), 'a+') as f:
            f.write("[%d/%d]\tAcc_Test: %.5f%%\n" %
                    (epoch, args.epoch, test_acc))