def set_jurkat_configs(args):
    args.batch_size = 256
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 1024
    args.stop_labeled = 1000
    args.add_labeled = 100
    args.merged = False
    args.autoencoder_z_dim = 128

    return args
