import os
import os.path as osp
import shutil
import warnings
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate

from mmhuman3d.apis import (
    feature_extract,
    inference_image_based_model,
    inference_video_based_model,
    init_model,
)
from mmhuman3d.core.visualization.visualize_smpl import render_smpl, visualize_smpl_hmr
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.utils.demo_utils import prepare_frames
from mmhuman3d.utils.ffmpeg_utils import array_to_images
from mmhuman3d.utils.transforms import rotmat_to_aa

def diff_render_test(args,frames_iter):
    
    assert args.mesh_reg_config == "configs/sshmr/sshmr.py"

    model, _ = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())
    
    device = next(model.parameters()).device

    batch_data = []

    for i,img in enumerate(frames_iter):
        img = torch.Tensor(img).permute(2,0,1).to(device)
        data = {
            'img':img,
            'img_metas':{},
            'sample_idx':i
        }
        # batch_data['img'].append(img)
        # batch_data['img_metas'].append(i)
        # batch_data['sample_idx'].append(i)
        batch_data.append(data)

    batch_size = len(batch_data)
    batch_data = collate(batch_data, samples_per_gpu=batch_size)

    results = model(
        img=batch_data['img'],
        img_metas=batch_data['img_metas'],
        sample_idx=batch_data['sample_idx'],
    )

    for k,v in results.items():
        if isinstance(v,torch.Tensor):
            print(f"{k} shape:{v.shape} requires_grad:{v.requires_grad}")

    dtype = results['vertices'].dtype

    focal_length = 5000
    img_res = 244
    camera_center = torch.zeros([batch_size, 2])

    R = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)
    
    T = torch.stack([
        results['camera'][:, 1], results['camera'][:, 2], 2 * focal_length /
        (img_res * results['camera'][:, 0] + 1e-9)
    ],dim=-1)

    K = torch.zeros([batch_size, 3, 3], device=device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    K = K.to(device)
    R = R.to(device)
    T = T.to(device)

    body_model_config = dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy')

    body_model = build_body_model(body_model_config).to(device)

    render_tensor = render_smpl(
        verts = results['vertices'],
        body_model=body_model,
        K = K,
        R = R,
        T = T,
        render_choice = 'hq',
        palette = 'segmentation',
        resolution = img_res,
        return_tensor = True,
        image_array = batch_data['img']
    )

    print(f"{render_tensor} shape:{render_tensor.shape} requires_grad:{render_tensor.requires_grad}")

def main(args):

    # prepare input
    frames_iter = prepare_frames(args.input_path)

    if args.custom_process:
        diff_render_test(args,frames_iter)
    else:
        raise ValueError(
            'Only supports single_person_demo or multi_person_demo')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        'mesh_reg_config',
        type=str,
        default=None,
        help='Config file for mesh regression')
    parser.add_argument(
        'mesh_reg_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for mesh regression')
    parser.add_argument(
        '--single_person_demo',
        action='store_true',
        help='Single person demo with MMDetection')
    parser.add_argument('--det_config', help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--multi_person_demo',
        action='store_true',
        help='Multi person demo with MMTracking')
    parser.add_argument('--tracking_config', help='Config file for tracking')

    parser.add_argument(
        '--body_model_dir',
        type=str,
        default='data/body_models/',
        help='Body models file path')
    parser.add_argument(
        '--input_path', type=str, default=None, help='Input path')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='directory to save output result file')
    parser.add_argument(
        '--show_path',
        type=str,
        default=None,
        help='directory to save rendered images or video')
    parser.add_argument(
        '--render_choice',
        type=str,
        default='hq',
        help='Render choice parameters')
    parser.add_argument(
        '--palette', type=str, default='segmentation', help='Color theme')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.99,
        help='Bounding box score threshold')
    parser.add_argument(
        '--draw_bbox',
        action='store_true',
        help='Draw a bbox for each detected instance')
    parser.add_argument(
        '--smooth_type',
        type=str,
        default=None,
        help='Smooth the data through the specified type.'
        'Select in [oneeuro,gaus1d,savgol].')
    parser.add_argument(
        '--speed_up_type',
        type=str,
        default=None,
        help='Speed up data processing through the specified type.'
        'Select in [deciwatch].')
    parser.add_argument(
        '--focal_length', type=float, default=5000., help='Focal lenght')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    parser.add_argument(
        '--custom_process',
        type=bool,
        default=True,
        help='run custom process')
    args = parser.parse_args()
    main(args)

"""
python demo/estimate_smpl_copy.py \
    configs/sshmr/sshmr.py \
    data/checkpoints/resnet50_hmr_pw3d.pth \
    --single_person_demo \
    --det_config demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --input_path  demo/resources/image \
    --show_path vis_results/demo_image/ \
    --output demo_result \
    --smooth_type savgol \
    --speed_up_type deciwatch \
    --draw_bbox
"""