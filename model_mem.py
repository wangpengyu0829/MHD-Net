from models import resnet_mem # 无对比



def generate_model(opt):
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False

    assert opt.model_name in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']

    if opt.model_name == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet_mem.resnet10(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 18:
            model = resnet_mem.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 34:
            model = resnet_mem.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 50:
            model = resnet_mem.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 101:
            model = resnet_mem.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 152:
            model = resnet_mem.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 200:
            model = resnet_mem.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)

    return model




