# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from .expressive_mesh_estimator import SMPLXImageBodyModelEstimator
from .hybrik import HybrIK_trainer
from .custom_mesh_estimator import CustomImageBodyModelEstimator
from .mesh_estimator import ImageBodyModelEstimator, VideoBodyModelEstimator
from .self_sup_mesh_estimator import SelfSupImageBodyModelEstimator


def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    return MMCV_MODELS.build_func(cfg, registry, default_args)


ARCHITECTURES = Registry(
    'architectures', parent=MMCV_MODELS, build_func=build_from_cfg)

ARCHITECTURES.register_module(name='HybrIK_trainer', module=HybrIK_trainer)
ARCHITECTURES.register_module(
    name='ImageBodyModelEstimator', module=ImageBodyModelEstimator)
ARCHITECTURES.register_module(
    name='VideoBodyModelEstimator', module=VideoBodyModelEstimator)
ARCHITECTURES.register_module(
    name='SMPLXImageBodyModelEstimator', module=SMPLXImageBodyModelEstimator)
ARCHITECTURES.register_module(
    name='CustomImageBodyModelEstimator', module=CustomImageBodyModelEstimator)
ARCHITECTURES.register_module(
    name='SelfSupImageBodyModelEstimator', module=SelfSupImageBodyModelEstimator)


def build_architecture(cfg):
    """Build framework."""
    return ARCHITECTURES.build(cfg)
