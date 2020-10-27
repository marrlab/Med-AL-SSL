def set_isic_configs(args):
    args.batch_size = 128
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 768
    args.stop_labeled = 20250
    args.add_labeled = 100
    args.start_labeled = 20100
    args.merged = False

    if args.novel_class_detection:
        args.remove_classes = True

    return args
