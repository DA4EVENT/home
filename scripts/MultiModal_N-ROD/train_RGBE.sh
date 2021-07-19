
set -ex

###############
# Train Param #
###############

# voxelgrid9 offline
# remember to add the paths of the dataset in utils.py


L=$1 #LR
F=$2 #file.py
BS=$3 #BS
Name=$4 #name_exp
MVnet=$5 # MVnet or null
Rot=$6 #weight rot solo 0.0
Ent=$7 #weight entropy 0.0
Disc=$8 #weight GRL 0.0
MMD=${9} #weight MMD
AFN=${10} #weight AFN
LR_EVENT_Mult=${11} #weight lr of event respect to RGB net
FC_mult=${12} #fc mult

$Project_path=<paths>/Vision_Event_DA ## add path of the project

###############################
#   RGB-E (RGB-Voxelgrid9)    #
###############################

python3 $Project_path/N-ROD/$F \
--dataset ROD \
--source Syn \
--target Real \
--source_data_format rgb-evrepr \
--target_data_format rgb-evrepr \
--modality rgb-voxelgrid_9chans \
--task $Name \
--net resnet18 \
--epoch 30 \
--lr $L \
--lr_mult $FC_mult \
--lr_event_mult $LR_EVENT_Mult \
--batch_size $BS \
--weight_decay 0.0001 \
--weight_rot $Rot \
--weight_ent $Ent \
--weight_grl $Disc \
--weight_mmd $MMD \
--weight_afn $AFN \
--$MVnet \

