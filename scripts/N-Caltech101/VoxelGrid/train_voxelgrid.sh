
set -ex


L=$1 #LR
F=$2 #file.py
Name=$3 #name
MVnet=$4 # MVnet or null
Rot=$5 #weight rot solo 0.0
Ent=$6 #weight entropy 0.0
Disc=$7 #weight GRL 0.0
MMD=${8} #weight MMD
AFN=${9} #weight AFN


#####################################
#   Voxelgrid 9 channels Offline    #
#####################################

$Project_path=<paths>/Vision_Event_DA ## add path of the project


python3 $Project_path/N-Caltech101/$F \
--dataset ncaltech101 \
--source Sim \
--target Real \
--source_data_format evrepr \
--target_data_format evrepr \
--modality voxelgrid_9chans \
--class_num 101 \
--task $Name \
--gpu 0 \
--net resnet34 \
--epoch 30 \
--lr $L \
--batch_size 32 \
--weight_decay 0.0001 \
--weight_rot $Rot \
--weight_ent $Ent \
--weight_grl $Disc \
--weight_mmd $MMD \
--weight_afn $AFN \
--GB_class \
--$MVnet \



### to use esim online + voxelgrid representation, add the following params
#--source_data_format event_images \
#--esim_threshold_range 0.05 0.05 \
#--esim_use_log False \
#--target_data_format event_bin \
#--evrepr RPGVoxelGrid \
#--rpgvoxelgrid_bins 9 \ #number of channels
#--evrepr_trainable False \
#--evrepr_frame_size 180 240 \


