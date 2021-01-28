def set_matek_configs(args):
    args.batch_size = 128
    args.fixmatch_k_img = 8192
    args.simclr_batch_size = 768
    args.stop_labeled = 6615
    args.add_labeled = 735
    args.start_labeled = 147
    args.pseudo_labeling_num = 14692 - args.stop_labeled

    if args.novel_class_detection:
        args.remove_classes = True

    return args
