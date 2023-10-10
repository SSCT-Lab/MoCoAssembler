# encoding: utf-8
from .m2net import M2Net

def build_model_mindspore(cfg, num_classes, num_apps, training):
    model = M2Net(num_classes, num_apps, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NAME,
                     cfg.MODEL.POOLING_TYPE, cfg.MODEL.PRETRAIN_CHOICE, cfg.MODEL.CROSSMODEITY, training)
    return model