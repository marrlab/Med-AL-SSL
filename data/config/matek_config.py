def set_matek_configs(args):
    args.batch_size = 256
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 512

    return args
