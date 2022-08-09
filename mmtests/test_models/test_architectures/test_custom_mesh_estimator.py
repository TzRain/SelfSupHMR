import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmhuman3d import __version__
from mmhuman3d.apis import set_random_seed, train_model
from mmhuman3d.data.datasets import build_dataset
from mmhuman3d.models.architectures.builder import build_architecture
from mmhuman3d.utils.collect_env import collect_env
from mmhuman3d.utils.logger import get_root_logger

import torch.distributed as dist

def test_custom_mesh_estimator():
    
    config_file = 'configs/custom/resnet50_hmr_render.py'
    cfg = Config.fromfile(config_file)
    model = build_architecture(cfg.model)
    if cfg.model.type == 'CustomImageBodyModelEstimator':
        with open("./work_dirs/custom_render/custom_mesh_emstimator_model_namespace.txt","w") as f:
            for name, params in model.named_parameters():
                print(name, file=f)

        check_point_path = 'data/checkpoints/resnet50_hmr_pw3d.pth'
        if os.path.isfile(check_point_path):
            pretrained_state_dict = torch.load(check_point_path,map_location=lambda storage, loc: storage)
        
        with open("./work_dirs/custom_render/resnet50_hmr_pw3d_namespace.txt","w") as f:
            for name, params in pretrained_state_dict['state_dict'].items():
                print(name, file=f)
        
        model.load_state_dict(pretrained_state_dict['state_dict'],strict=False)
        print('load pretrain model for CustomImageBodyModelEstimator')
        
    else: 
        raise Exception('It is not CustomImageBodyModelEstimator!!!')
    
    return model

if __name__ == '__main__':
    print(test_custom_mesh_estimator())