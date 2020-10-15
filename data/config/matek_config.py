def set_matek_configs(args):
    args.batch_size = 128
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 768
    args.stop_labeled = 1000
    args.add_labeled = 100
    args.autoencoder_z_dim = 128

    return args
