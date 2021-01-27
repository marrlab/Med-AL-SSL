def set_retinopathy_configs(args):
    args.batch_size = 128
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 768
    args.stop_labeled = 1647
    args.add_labeled = 183
    args.start_labeled = 37
    args.merged = False
    args.pseudo_labeling_num = 2929 - args.stop_labeled

    if args.novel_class_detection:
        args.remove_classes = True

    return args
