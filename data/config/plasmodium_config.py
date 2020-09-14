def set_plasmodium_configs(args):
    args.batch_size = 512
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 1024
    args.labeled_ratio_start = 0.025
    args.labeled_ratio_stop = 0.225
    args.oversampling = False

    return args
