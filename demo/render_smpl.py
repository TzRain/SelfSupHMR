import os
import os.path as osp
import shutil
import warnings
from argparse import ArgumentParser
from pathlib import Path
from mmcv.parallel import collate
from mmhuman3d.apis.inference import LoadImage
from mmhuman3d.utils.demo_utils import box2cs, xywh2xyxy, xyxy2xywh

import mmcv
import numpy as np
import torch
from mmhuman3d.data.datasets.pipelines import Compose
from mmhuman3d.apis import (
    feature_extract,
)
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_hmr
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.architectures.custom_mesh_estimator import save_img
from mmhuman3d.utils.demo_utils import (
    extract_feature_sequence,
    get_speed_up_interval,
    prepare_frames,
    process_mmdet_results,
    process_mmtracking_results,
    smooth_process,
    speed_up_interpolate,
    speed_up_process,
)
from mmhuman3d.utils.ffmpeg_utils import array_to_images
from mmhuman3d.utils.transforms import rotmat_to_aa
from mmtests.test_models.test_architectures.test_custom_mesh_estimator import test_custom_mesh_estimator

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    from mmtrack.apis import inference_mot
    from mmtrack.apis import init_model as init_tracking_model
    has_mmtrack = True
except (ImportError, ModuleNotFoundError):
    has_mmtrack = False



def custom_renderer(results):
    smpl_poses = results['smpl_pose']
    smpl_betas = results['smpl_beta']
    pred_cams = results['camera']
    affined_img = results['affined_img']

    print("run custom renderer")

    smpl_poses = np.array(smpl_poses)
    smpl_betas = np.array(smpl_betas)
    pred_cams = np.array(pred_cams)
    affined_img = np.array(affined_img)

    if smpl_poses.shape[1:] == (24, 3, 3):
        smpl_poses = rotmat_to_aa(smpl_poses)
    

    body_model_config = dict(model_path="data/body_models/", type='smpl')
    tensors = visualize_smpl_hmr(
        poses=smpl_poses.reshape(-1, 24 * 3),
        betas=smpl_betas,
        cam_transl=pred_cams,
        output_path='vis_results/custom_demo',
        render_choice='hq',
        resolution=affined_img[0].shape[:2],
        image_array=affined_img,
        body_model_config=body_model_config,
        overwrite=True,
        return_tensor = True,
        no_grad = False,
        palette='segmentation',
        read_frames_batch=True)
    
    save_img(tensors,'demo')

        


def get_detection_result(args, frames_iter, mesh_model, extractor):
    
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    frame_id_list = []
    result_list = []
    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        mmdet_results = inference_detector(person_det_model, frame)
        # keep the person class bounding boxes.
        results = process_mmdet_results(
            mmdet_results, cat_id=args.det_cat_id, bbox_thr=args.bbox_thr)

        # extract features from the input video or image sequences
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator' \
                and extractor is not None:
            results = feature_extract(
                extractor, frame, results, args.bbox_thr, format='xyxy')
        # drop the frame with no detected results
        if results == []:
            continue
        # vis bboxes
        if args.draw_bbox:
            bboxes = [res['bbox'] for res in results]
            bboxes = np.vstack(bboxes)
            mmcv.imshow_bboxes(
                frame, bboxes, top_k=-1, thickness=2, show=False)
        frame_id_list.append(i)
        result_list.append(results)

    return frame_id_list, result_list

def inference_image_based_model(
    model,
    img_or_path,
    det_results,
    bbox_thr=None,
    format='xywh',
):
    """Inference a single image with a list of person bounding boxes.

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (Union[str, np.ndarray]): Image filename or loaded image.
        det_results (List(dict)): the item in the dict may contain
            'bbox' and/or 'track_id'.
            'bbox' (4, ) or (5, ): The person bounding box, which contains
            4 box coordinates (and score).
            'track_id' (int): The unique id for each human instance.
        bbox_thr (float, optional): Threshold for bounding boxes.
            Only bboxes with higher scores will be fed into the pose detector.
            If bbox_thr is None, ignore it. Defaults to None.
        format (str, optional): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).

    Returns:
        list[dict]: Each item in the list is a dictionary,
            containing the bbox: (left, top, right, bottom, [score]),
            SMPL parameters, vertices, kp3d, and camera.
    """
    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']
    if len(det_results) == 0:
        return []

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in det_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        det_results = [det_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xywh = xyxy2xywh(bboxes)
    else:
        bboxes_xywh = bboxes

    if len(bboxes_xywh) == 0:
        return []

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    inference_pipeline = [LoadImage()] + cfg.inference_pipeline
    inference_pipeline = Compose(inference_pipeline)

    assert len(bboxes[0]) in [4, 5]

    batch_data = []
    input_size = cfg['img_res']
    aspect_ratio = 1 if isinstance(input_size, int) else input_size[0] / input_size[1]

    for i, bbox in enumerate(bboxes_xywh):
        center, scale = box2cs(bbox, aspect_ratio, bbox_scale_factor=1.25)
        # prepare data
        data = {
            'image_path': img_or_path,
            'center': center,
            'scale': scale,
            'rotation': 0,
            'bbox_score': bbox[4] if len(bbox) == 5 else 1,
            'sample_idx': i,
        }
        data = inference_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(device)

    # get all img_metas of each bounding box
    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]

    # forward the model
    with torch.no_grad():
        results = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            sample_idx=batch_data['sample_idx'],
        )
    return results

def single_person_with_mmdet(args,load_human_data=False):
    """Estimate smpl parameters from single-person
        images with mmdetection
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames

    """
    frames_iter = prepare_frames(args.input_path)

    extractor = None 
    mesh_model = test_custom_mesh_estimator()
    mesh_model.cfg = mmcv.Config.fromfile(args.mesh_reg_config) 
    mesh_model.to(args.device.lower())
    mesh_model.eval()

    # mesh_model, extractor = init_model(
    #     args.mesh_reg_config,
    #     args.mesh_reg_checkpoint,
    #     device=args.device.lower())

    pred_cams, verts, smpl_poses, smpl_betas, bboxes_xyxy, affined_img= \
        [], [], [], [], [], []

    frame_id_list, result_list = \
        get_detection_result(args, frames_iter, mesh_model, extractor)

    for i, result in enumerate(mmcv.track_iter_progress(result_list)):
        frame_id = frame_id_list[i]
        inference_image_based_model(
            mesh_model,
            frames_iter[frame_id],
            result,
            bbox_thr=args.bbox_thr,
            format='xyxy')
        
    # release GPU memory
    del mesh_model
    del extractor
    torch.cuda.empty_cache()


def main(args):

    # prepare input
    

    if args.single_person_demo:
        single_person_with_mmdet(args)


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
    args = parser.parse_args()

    if args.single_person_demo:
        assert has_mmdet, 'Please install mmdet to run the demo.'
        assert args.det_config is not None
        assert args.det_checkpoint is not None

    if args.multi_person_demo:
        assert has_mmtrack, 'Please install mmtrack to run the demo.'
        assert args.tracking_config is not None

    main(args)
