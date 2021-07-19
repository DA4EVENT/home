
#L=$1 #LR
#F=$2 #file.py
#Name=$3 #name
#MVnet=$4 # MVnet or null
#Rot=$5 #weight rot solo 0.0
#Ent=$6 #weight entropy 0.0
#Disc=$7 #weight GRL 0.0
#MMD=${8} #weight MMD
#AFN=${9} #weight AFN


# Domain shift: Sim-->Real
# Voxelgrid 9 channels

###############
# Source Only #
###############

sh train_voxelgrid.sh 0.001 train_single_modality.py  SourceOnly null  0.0 0.0 0.0 0.0 0.0
sh train_voxelgrid.sh 0.001 train_single_modality.py  MVnet_SourceOnly AvgChannels  0.0 0.0 0.0 0.0 0.0

################
#      GRL     #
################

sh train_voxelgrid.sh 0.001 train_GRL.py  GRL null  0.0 0.0 1.0 0.0 0.0
sh train_voxelgrid.sh 0.001 train_GRL.py  MVnet_GRL AvgChannels  0.0 0.0 1.0 0.0 0.0

################
#      MMD     #
################

sh train_voxelgrid.sh 0.001 train_MMD.py MMD null  0.0 0.0 0.0 1.0 0.0
sh train_voxelgrid.sh 0.001 train_MMD.py MVnet_MMD AvgChannels 0.0 0.0 0.0 1.0 0.0

####################
#      Entropy     #
####################

sh train_voxelgrid.sh 0.001 train_single_modality.py Entropy null 0.0 0.1 0.0 0.0 0.0
sh train_voxelgrid.sh 0.001 train_single_modality.py MVnet_Entropy AvgChannels 0.0 0.1 0.0 0.0 0.0

################
#      AFN     #
################

sh train_voxelgrid.sh 0.001 train_AFN.py AFN null 0.0 0.01 0.0 0.0 0.01
sh train_voxelgrid.sh 0.001 train_AFN.py MVnet_AFN AvgChannels 0.0 0.01 0.0 0.0 0.01

################
#      Rot     #
################

sh train_voxelgrid.sh 0.001 train_single_modality.py Standard_Rotation null 1.0 0.0 0.0 0.0 0.0
sh train_voxelgrid.sh 0.001 train_single_modality.py MVnet_Standard_Rotation AvgChannels 1.0 0.0 0.0 0.0 0.0

