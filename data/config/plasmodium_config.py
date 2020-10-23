def set_plasmodium_configs(args):
    args.batch_size = 256
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 2048
    args.stop_labeled = 661
    args.add_labeled = 110
    args.start_labeled = 110
    args.merged = False

    if args.novel_class_detection:
        args.remove_classes = True

    return args
