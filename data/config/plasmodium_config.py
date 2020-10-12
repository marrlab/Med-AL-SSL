def set_plasmodium_configs(args):
    args.batch_size = 256
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 2048
    args.labeled_stop = 1000
    args.add_labeled = 100
    args.remove_classes = False
    args.merged = False

    return args
