def set_jurkat_configs(args):
    args.batch_size = 1024
    args.fixmatch_k_img = 16384
    args.simclr_batch_size = 1024
    args.labeled_ratio_start = 0.01
    args.labeled_ratio_stop = 0.2

    return args
