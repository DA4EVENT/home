#!/bin/bash




LR=$1 #LR
File=$2 #file.py
Name=$3 #name
MVnet=$4 # MVnet or Null
Rot=$5 #weight rot
Ent=$6 #weight entropy
Disc=$7 #weight GRL
MMD=$8 #weight MMD
AFN=$9 #weight AFN


set -ex


#####################################
#      EST 9 channels Offline       #
#####################################


$Project_path=<paths>/Vision_Event_DA ## add path of the project


# ESIM online
python3 $Project_path/Vision_Event_DA/N-Caltech101/$File \
--dataset ncaltech101 \
--source Sim \
--target Real \
--source_data_format event_images \
--target_data_format event_bin \
--esim_threshold_range 0.16 0.16 \
--source_subsample_mode absolute \
--source_subsample_value 700000 \
--esim_use_log False \
--evrepr EST \
--evrepr_trainable True \
--evrepr_frame_size 180 240 \
--class_num 101 \
--task $Name \
--net resnet34 \
--epoch 30 \
--lr $LR \
--batch_size 32 \
--weight_decay 0.0001 \
--weight_rot $Rot \
--weight_ent $Ent \
--weight_grl $Disc \
--weight_afn $AFN \
--weight_mmd $MMD \
--$MVnet \
--GB_class