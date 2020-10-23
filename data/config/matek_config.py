def set_matek_configs(args):
    args.batch_size = 128
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 768
    args.stop_labeled = 2938
    args.add_labeled = 588
    args.start_labeled = 147

    if args.novel_class_detection:
        args.remove_classes = True

    return args
