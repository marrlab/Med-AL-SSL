def set_matek_configs(args):
    args.batch_size = 384
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 768
    args.labeled_ratio_start = 0.05
    args.labeled_ratio_stop = 0.25
    args.add_labeled_ratio = 0.025

    return args
