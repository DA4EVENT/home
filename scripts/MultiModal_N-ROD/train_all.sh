set -ex

#L=$1 #LR
#F=$2 #file.py
#BS=$3 #BS
#Name=$4 #name_exp
#MVnet=$5 # MVnet or null
#Rot=$6 #weight rot solo 0.0
#Ent=$7 #weight entropy 0.0
#Disc=$8 #weight GRL 0.0
#MMD=${9} #weight MMD
#AFN=${10} #weight AFN
#LR_EVENT_Mult=${11} #lr-multiply of event-net w.r.t. RGB-net
#FC_mult=${12} #fc mult

###############
# Source Only #
###############

sh train_event.sh 0.0001 train_MultiModal.py 32 Source_Only null  0.0 0.0 0.0 0.0 0.0 1 1
sh train_event.sh 0.0001 train_MultiModal.py 32 MVnet_Source_Only AvgChannels  0.0 0.0 0.0 0.0 0.0 1 1

################
#      GRL     #
################

sh train_event.sh 0.0001 train_MM_GRL.py 32 GRL null 0.0 0.0 1.0 0.0 0.0 1 1
sh train_event.sh 0.0001 train_MM_GRL.py 32 MVnet_GRL AvgChannels 0.0 0.0 1.0 0.0 0.0 1 1

################
#      MMD     #
################

sh train_event.sh 0.0001 train_MM_MMD.py 32 MMD null  0.0 0.0 0.0 1.0 0.0 1 1
sh train_event.sh 0.0001 train_MM_MMD.py 32 MVnet_MMD AvgChannels 0.0 0.0 0.0 1.0 0.0 1 1

####################
#      Entropy     #
####################

sh train_event.sh 0.0001 train_MultiModal.py 32 Entropy null 0.0 0.1 0.0 0.0 0.0 1 1
sh train_event.sh 0.0001 train_MultiModal.py 32 MVnet_Entropy AvgChannels 0.0 0.1 0.0 0.0 0.0 1 1

################
#      AFN     #
################

sh train_event.sh 0.0001 train_MM_AFN.py 32 AFN null 0.0 0.1 0.0 0.0 0.05 1 1
sh train_event.sh 0.0001 train_MM_AFN.py 32 MVnet_AFN AvgChannels 0.0 0.1 0.0 0.0 0.05 1 1

################
#      Rot     #
################

sh train_event.sh 0.0001 train_MM_RR.py 32 Relative_Rotation null 0.01 0.0 0.0 0.0 0.0 1 1
sh train_event.sh 0.0001 train_MM_RR.py 32 MVnet_Relative_Rotation AvgChannels 0.01 0.0 0.0 0.0 0.0 1 1
