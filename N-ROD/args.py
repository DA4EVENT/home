import argparse


def arg_boolean(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_base_args(parser: argparse.ArgumentParser):
    # Dataset arguments

    #Caltech - Cifar
    parser.add_argument('--dataset', default='', choices=['cifar10', 'caltech101','ncaltech101', 'ROD'])
    parser.add_argument('--target', default='', choices=['Real', 'Syn', 'Sim'])
    parser.add_argument('--source', default='', choices=['Real', 'Syn', 'Sim'])
    parser.add_argument('--target_data_format', default='evrepr',
                        choices=['evrepr', 'rgb', "depth", "rgb-depth", 'event_bin', 'event_aedat', 'event_images', 'event_dat',
                                 'rgb-evrepr', 'rgb-event_bin', 'rgb-event_aedat', 'event_npz', 'rgb-event_images', 'rgb-event_dat'])
    parser.add_argument('--source_data_format', default='evrepr',
                        choices=['evrepr', 'rgb', "depth","rgb-depth", 'event_bin', 'event_aedat', 'event_images', 'event_dat',
                                 'rgb-evrepr', 'rgb-event_bin', 'rgb-event_aedat', 'event_npz', 'rgb-event_images', 'rgb-event_dat'])
    parser.add_argument("--modality", default="", choices=[
        'rgb', "hats", "depth", "rgb-depth",
        'eventvolume', 'voxelgrid_3chans', 'voxelgrid_6chans', 'voxelgrid_9chans',
        'rgb-eventvolume', 'rgb-voxelgrid_3chans', 'rgb-voxelgrid_6chans', 'rgb-voxelgrid_9chans',
    ])

    parser.add_argument("--data_root_source", default=None)
    parser.add_argument("--data_root_target", default=None)
    parser.add_argument("--train_file_source", default=None)
    parser.add_argument("--test_file_source", default=None)
    parser.add_argument("--train_file_target", default=None)
    parser.add_argument("--test_file_target", default=None)
    parser.add_argument("--class_num", default=1000, type=int)

    parser.add_argument("--task", default="DA4Event")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--snapshot", default="./tensorboard/snapshot/")
    parser.add_argument("--experiment", default="./tensorboard")
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('--suffix', default="")

    # hyper-params
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--epoch", default=40, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--lr_mult", default=1.0, type=float)
    parser.add_argument("--lr_event_mult", default=1.0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_accumulation", default=1, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--dropout_p", default=0.5)


    # DA methods
    parser.add_argument("--weight_rot", default=0.0, type=float)
    parser.add_argument('--weight_ent', default=0.0, type=float)
    parser.add_argument('--weight_grl', default=0.0, type=float)
    parser.add_argument('--weight_mmd', default=0.0, type=float)
    parser.add_argument('--weight_afn', default=0.0, type=float)

    # Event
    parser.add_argument("--channels_event", default=15, type=int)
    parser.add_argument("--AvgChannels", action='store_true')
    parser.add_argument("--Adam", action='store_true')
    parser.add_argument("--representation_type", default="")
    parser.add_argument("--GB_class", action='store_true')
    parser.add_argument("--imagenet_norm", action='store_true')
    parser.add_argument("--NPZ", action='store_true')



    parser.add_argument("--null", action='store_true')
    parser.add_argument("--SaveModel", action='store_true')
    parser.add_argument("--est_lr_mul", default=1, type=int)

    # Event subsample
    parser.add_argument("--source_subsample_mode", default=None,
                        choices=["absolute", "relative"])
    parser.add_argument("--source_subsample_threshold", type=int, default=None)
    parser.add_argument("--source_subsample_value", type=float, default=500e3)

    parser.add_argument("--target_subsample_mode", default=None,
                        choices=["absolute", "relative"])
    parser.add_argument("--target_subsample_threshold", type=int, default=None)
    parser.add_argument("--target_subsample_value", type=float, default=500e3)

    # Event Representation Args
    parser.add_argument('--evrepr', default=None,
                        choices=["CountsLastTs", "E2Vid", "EST",
                                 "EventVolume", "RPGVoxelGrid",
                                 "MatrixLSTM", "HATS", "TBR"])
    parser.add_argument('--evrepr_trainable', type=arg_boolean, default=False)
    parser.add_argument('--evrepr_crop', type=arg_boolean, default=True)
    parser.add_argument('--evrepr_frame_size', type=int, nargs='+', default=[180, 240])

    # CountsLastTs args
    parser.add_argument("--countlastts_bins", type=int, default=1)
    parser.add_argument("--countlastts_features", type=int,
                        nargs='+', default=['counts', 'last_ts'])

    # EventVolume args
    parser.add_argument("--eventvolume_bins", type=int, default=9)

    # RPGVoxelGrid args
    parser.add_argument("--rpgvoxelgrid_bins", type=int, default=9)

    # E2Vid args
    parser.add_argument("--e2vid_weights", type=str,
                        default="http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar")
    parser.add_argument("--e2vid_arch", type=str, default="e2vidrecurrent")
    parser.add_argument("--e2vid_bins", type=int, default=5)
    parser.add_argument("--e2vid_skip_type", type=str, default="sum")
    parser.add_argument("--e2vid_num_encoders", type=int, default=4)
    parser.add_argument("--e2vid_base_num_channels", type=int, default=32)
    parser.add_argument("--e2vid_num_residual_blocks", type=int, default=2)
    parser.add_argument("--e2vid_norm", type=str, default=None)
    parser.add_argument("--e2vid_use_upsample_conv", type=arg_boolean, default=True)

    # EST args
    parser.add_argument("--est_bins", type=int, default=9)
    parser.add_argument("--est_mlp_layers", type=int, nargs='+', default=[1, 30, 30, 1])
    parser.add_argument("--est_activation", type=str, default="LeakyReLU")
    parser.add_argument("--est_activation_args", type=dict, default={"negative_slope": 0.1})

    # MatrixLSTM args
    parser.add_argument("--matrixlstm_region_shape", type=int, nargs='+', default=[1, 1])
    parser.add_argument("--matrixlstm_region_stride", type=int, nargs='+', default=[1, 1])
    parser.add_argument("--matrixlstm_input_size", type=int, default=1)
    parser.add_argument("--matrixlstm_hidden_size", type=int, default=3)
    parser.add_argument("--matrixlstm_num_layers", type=int, default=1)
    parser.add_argument("--matrixlstm_bias", type=arg_boolean, default=True)
    parser.add_argument("--matrixlstm_lstm_type", type=str, default="LSTM")
    parser.add_argument("--matrixlstm_add_coords_features", type=arg_boolean, default=False)
    parser.add_argument("--matrixlstm_add_feature_mode", type=str, default="delay_norm")
    parser.add_argument("--matrixlstm_normalize_relative", type=arg_boolean, default=True)
    parser.add_argument("--matrixlstm_max_events_per_rf", type=int, default=128)
    parser.add_argument("--matrixlstm_maintain_in_shape", type=arg_boolean, default=True)
    parser.add_argument("--matrixlstm_keep_most_recent", type=arg_boolean, default=True)
    parser.add_argument("--matrixlstm_frame_intervals", type=int, default=1)
    parser.add_argument("--matrixlstm_frame_intervals_mode", type=str, default=None)
    parser.add_argument("--matrixlstm_add_selayer", type=arg_boolean, default=False)

    # HATS
    parser.add_argument("--hats_bins", type=int, default=2)
    parser.add_argument("--hats_r", type=int, default=3)
    parser.add_argument("--hats_k", type=int, default=10)
    parser.add_argument("--hats_tau", type=float, default=1e6)
    parser.add_argument("--hats_delta_t", type=float, default=100e3)
    parser.add_argument("--hats_minibatch", type=int, default=4)

    # ESIM args
    parser.add_argument("--esim_threshold_range", type=float, nargs='+', default=[0.06, 0.06]) #il nostro C
    parser.add_argument("--esim_refractory_period", type=float, default=1e6)
    parser.add_argument("--esim_log_eps", type=float, default=0.001)
    parser.add_argument("--esim_use_log", type=arg_boolean, default=False)
    parser.add_argument("--esim_timestamps_path", type=str, default=None)

    # TBR args
    parser.add_argument("--tbr_bins", type=int, default=8)
