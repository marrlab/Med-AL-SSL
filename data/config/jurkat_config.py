def set_jurkat_configs(args):
    args.batch_size = 256
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 1024
    args.stop_labeled = 5168
    args.add_labeled = 1033
    args.start_labeled = 258
    args.merged = False

    if args.novel_class_detection:
        args.remove_classes = True

    return args
