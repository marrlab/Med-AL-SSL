def set_jurkat_configs(args):
    args.batch_size = 1024
    args.fixmatch_k_img = 16384
    args.simclr_batch_size = 1024

    return args
