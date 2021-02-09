def set_jurkat_configs(args):
    args.batch_size = 256
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 896
    args.stop_labeled = 11609
    args.add_labeled = 1290
    args.start_labeled = 258
    args.merged = False
    args.pseudo_labeling_num = 25481 - args.stop_labeled

    if args.novel_class_detection:
        args.remove_classes = True

    return args
