# encoding: utf-8
"""
@author:  chenjiayang
@contact: sychenjiayang@163.com
"""
from PIL import Image
#import torchvision.transforms as T
from . import cla_transforms as CT

def build_transforms(dataset_type, cfg, is_train=True):
    if dataset_type == "sequence-classification":
        transforms = build_transforms_sequence_classification(cfg, is_train=is_train)
    elif dataset_type == "none":
        transforms = None
    else:
        raise Exception("Wrong Transforms Type!")
    return transforms


def build_transforms_sequence_classification(cfg, is_train=True):
    #crop_length = 2000  #777
    if is_train:
        transform = CT.Compose([
            CT.ClampLength(min=0*cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE, max=50 * cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE),
            #CT.RandomCrop(cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE),
            #CT.Encode(),
            CT.ToTensor(),
        ])
    else:
        transform = CT.Compose([
            CT.ClampLength(min=0*cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE, max=50 * cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE),
            #CT.RandomCrop(cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE),
            #CT.Encode(),
            CT.ToTensor(),
        ])

    return transform
