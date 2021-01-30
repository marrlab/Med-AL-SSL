def set_isic_configs(args):
    args.batch_size = 128
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 512
    args.stop_labeled = 9134
    args.add_labeled = 1015
    args.start_labeled = 203
    args.merged = False
    args.pseudo_labeling_num = 20264 - args.stop_labeled

    if args.novel_class_detection:
        args.remove_classes = True

    return args
