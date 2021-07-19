set -ex

###############
# Train Param #
###############

# voxelgrid9 offline
# remember to add the paths of the dataset in utils.py

L=$1 #LR
F=$2 #file.py
G=$3 #batch size
Name=$4 #name_exp
MVnet=$5 # MVnet or null
Rot=$6 #weight rot solo 0.0
Ent=$7 #weight entropy 0.0
Disc=$8 #weight GRL 0.0
MMD=${9} #weight MMD
AFN=${10} #weight AFN
FC_mult=${11} #weight fc_mult

$Project_path=<paths>/Vision_Event_DA ## add path of the project


python3 $Project_path/N-ROD/$F \
--dataset ROD \
--source Syn \
--target Real \
--source_data_format evrepr \
--target_data_format evrepr \
--modality voxelgrid_9chans \
--task $Name \
--gpu $G \
--net resnet18 \
--epoch 30 \
--lr $L \
--lr_mult $FC_mult \
--batch_size 32 \
--weight_decay 0.0001 \
--weight_rot $Rot \
--weight_ent $Ent \
--weight_grl $Disc \
--weight_mmd $MMD \
--weight_afn $AFN \
--num_workers 4 \
--$NORM \
--$MVnet \


