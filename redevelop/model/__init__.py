# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

from .baseline import Baseline

def build_model(cfg):
    model = Baseline(
        backbone_name=cfg.MODEL.BACKBONE_NAME,
        seqwise_predictor_name=cfg.MODEL.SEQWISE_PREDICTOR_NAME,
        elewise_predictor_name=cfg.MODEL.ELEWISE_PREDICTOR_NAME,
        pairwise_predictor_name=cfg.MODEL.PAIRWISE_PREDICTOR_NAME,
        backbone_frozen=cfg.MODEL.BACKBONE_FROZEN,
        backbone_random_initialization=cfg.MODEL.BACKBONE_RANDOM_INITIALIZATION,
    )
    return model
