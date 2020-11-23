def set_cifar_configs(args):
    args.batch_size = 512
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 1024
    args.labeled_start = 0.05
    args.stop_labeled = 0.25
    args.add_labeled = 0.025
    args.merged = False
    args.remove_classes = False
    args.autoencoder_z_dim = 32

    return args
