def set_plasmodium_configs(args):
    args.batch_size = 512
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 1024
    args.labeled_ratio_start = 0.005
    args.labeled_ratio_stop = 0.2

    return args
