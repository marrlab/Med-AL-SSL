def set_plasmodium_configs(args):
    args.batch_size = 256
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 2048
    args.labeled_ratio_start = 0.0025
    args.labeled_ratio_stop = 0.075
    args.add_labeled_ratio = 0.01
    args.remove_classes = False
    args.merged = False

    return args
