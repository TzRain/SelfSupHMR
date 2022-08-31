import os
import cv2
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

    focal_length = 5000
    img_res = 224
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

    # render_tensor = render_smpl(
    #     verts = results['vertices'],
    #     body_model=body_model,
    #     K = K,
    #     R = R,
    #     T = T,
    #     render_choice = 'hq',
    #     palette = 'segmentation',
    #     resolution = img_res,
    #     return_tensor = True,
    #     batch_size = batch_size,
    #     no_grad=False,
    #     image_array = batch_data['img'].permute(0,2,3,1)
    # )

    # new render wolk through

    smpl_poses = results['smpl_pose']
    smpl_betas = results['smpl_beta']
    pred_cams = results['smpl_beta']

    if smpl_poses.shape[1:] == (24, 3, 3):
        smpl_poses = rotmat_to_aa(smpl_poses)
    
    bboxes_xyxy = []
    for i in range(batch_size):
        bboxes_xyxy.append([0,0,img_res,img_res])
    
    bboxes_xyxy = np.array(bboxes_xyxy)

    render_tensor = visualize_smpl_hmr(
        poses=smpl_poses.reshape(-1, 24 * 3),
        betas=smpl_betas,
        cam_transl=pred_cams,
        bbox=bboxes_xyxy,
        render_choice='hq',
        resolution=img_res,
        image_array = batch_data['img'].permute(0,2,3,1),
        body_model=body_model,
        overwrite=True,
        no_grad = False,
        return_tensor = True,
        batch_size = batch_size,
        palette='segmentation',
        read_frames_batch=True)
    
    # new render wolk through

    render_tensor_de = render_tensor.detach().cpu().numpy() * 256

    for i,img in enumerate(render_tensor_de):
        cv2.imwrite(f"{args.show_path}{i}.jpg",img)

    print(f"{render_tensor} shape:{render_tensor.shape} requires_grad:{render_tensor.requires_grad}")

def main(args):
    # prepare input
    frames_iter = prepare_frames(args.input_path)
    diff_render_test(args,frames_iter)

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
        '--focal_length', type=float, default=5000., help='Focal lenght')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    main(args)

"""
python demo/estimate_smpl_custom_process.py \
    configs/sshmr/sshmr.py \
    data/checkpoints/resnet50_hmr_pw3d.pth \
    --input_path  demo/resources/image \
    --show_path vis_results/demo_image/ \
    --output demo_result
"""