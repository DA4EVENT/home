set -ex

#L=$1 #LR
#F=$2 #file.py
#G=$3 #batch size
#Name=$4 #name_exp
#MVnet=$5 # MVnet or null
#Rot=$6 #weight rot solo 0.0
#Ent=$7 #weight entropy 0.0
#Disc=$8 #weight GRL 0.0
#MMD=${9} #weight MMD
#AFN=${10} #weight AFN
#FC_mult=${11} #weight fc_mult

###############
# Source Only #
###############

sh train_event.sh 0.0001 train_single_modality.py 32 Source_Only null  0.0 0.0 0.0 0.0 0.0 1
sh train_event.sh 0.0001 train_single_modality.py 32 MVnet_Source_Only AvgChannels  0.0 0.0 0.0 0.0 0.0 1

################
#      GRL     #
################

sh train_event.sh 0.0001 train_GRL.py 32 GRL null 0.0 0.0 1.0 0.0 0.0 1
sh train_event.sh 0.0001 train_GRL.py 32 MVnet_GRL AvgChannels 0.0 0.0 1.0 0.0 0.0 1

################
#      MMD     #
################

sh train_event.sh 0.0001 train_MMD.py 32 MMD null  0.0 0.0 0.0 1.0 0.0 1
sh train_event.sh 0.0001 train_MMD.py 32 MVnet_MMD AvgChannels 0.0 0.0 0.0 1.0 0.0 1

####################
#      Entropy     #
####################

sh train_event.sh 0.0001 train_single_modality.py 32 Entropy null 0.0 0.1 0.0 0.0 0.0 1
sh train_event.sh 0.0001 train_single_modality.py 32 MVnet_Entropy AvgChannels 0.0 0.1 0.0 0.0 0.0 1

################
#      AFN     #
################

sh train_event.sh 0.0001 train_AFN.py 32 AFN null 0.0 0.1 0.0 0.0 0.05 1
sh train_event.sh 0.0001 train_AFN.py 32 MVnet_AFN AvgChannels 0.0 0.1 0.0 0.0 0.05 1

################
#      Rot     #
################

sh train_event.sh 0.0001 train_single_modality.py 32 Standard_Rotation null 0.01 0.0 0.0 0.0 0.0 1
sh train_event.sh 0.0001 train_single_modality.py 32 MVnet_Standard_Rotation AvgChannels 0.01 0.0 0.0 0.0 0.0 1
