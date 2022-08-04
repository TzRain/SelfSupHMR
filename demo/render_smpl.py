import os
import os.path as osp
import shutil
import warnings
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
import torch

from mmhuman3d.apis import (
    feature_extract,
    inference_image_based_model,
    inference_video_based_model,
    init_model,
)
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_hmr
from mmhuman3d.data.data_structures.human_data import HumanData
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


def single_person_with_mmdet(args,load_human_data=False):
    """Estimate smpl parameters from single-person
        images with mmdetection
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames

    """
    if load_human_data == False:
        frames_iter = prepare_frames(args.input_path)

        mesh_model, extractor = init_model(
            args.mesh_reg_config,
            args.mesh_reg_checkpoint,
            device=args.device.lower())

        pred_cams, verts, smpl_poses, smpl_betas, bboxes_xyxy = \
            [], [], [], [], []

        frame_id_list, result_list = \
            get_detection_result(args, frames_iter, mesh_model, extractor)

        for i, result in enumerate(mmcv.track_iter_progress(result_list)):
            frame_id = frame_id_list[i]
            mesh_results = inference_image_based_model(
                mesh_model,
                frames_iter[frame_id],
                result,
                bbox_thr=args.bbox_thr,
                format='xyxy')
            smpl_betas.append(mesh_results[0]['smpl_beta'])
            smpl_pose = mesh_results[0]['smpl_pose']
            smpl_poses.append(smpl_pose)
            pred_cams.append(mesh_results[0]['camera'])
            verts.append(mesh_results[0]['vertices'])
            bboxes_xyxy.append(mesh_results[0]['bbox'])

        smpl_poses = np.array(smpl_poses)
        smpl_betas = np.array(smpl_betas)
        pred_cams = np.array(pred_cams)
        verts = np.array(verts)
        bboxes_xyxy = np.array(bboxes_xyxy)

        # release GPU memory
        del mesh_model
        del extractor
        torch.cuda.empty_cache()
        if smpl_poses.shape[1:] == (24, 3, 3):
            smpl_poses = rotmat_to_aa(smpl_poses)
        elif smpl_poses.shape[1:] == (24, 3):
            smpl_poses = smpl_pose
        else:
            raise (f'Wrong shape of `smpl_pose`: {smpl_pose.shape}')
        
        render_data = {
            "smpl_poses":smpl_poses,
            "smpl_betas":smpl_betas,
            "pred_cams":pred_cams,
            "bboxes_xyxy":bboxes_xyxy,
            "frames_iter":frames_iter
        } 
        np.save("demo_result/render_data_result.npz",render_data)
        
    else :
        print("load data")
        render_data=np.load("demo_result/render_data_result.npz", allow_pickle=True)
        smpl_poses = render_data.item().get('smpl_poses')
        smpl_betas = render_data.item().get('smpl_betas')
        pred_cams = render_data.item().get('pred_cams')
        bboxes_xyxy = render_data.item().get('bboxes_xyxy')
        frames_iter = render_data.item().get('frames_iter')
        frame_id_list  = []
        for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
            frame_id_list.append(i)

    if args.output is not None and load_human_data == False:
        body_pose_, global_orient_, smpl_betas_, verts_, pred_cams_, \
            bboxes_xyxy_, image_path_, person_id_, frame_id_ = \
            [], [], [], [], [], [], [], [], []
        human_data = HumanData()
        frames_folder = osp.join(args.output, 'images')
        os.makedirs(frames_folder, exist_ok=True)
        array_to_images(
            np.array(frames_iter)[frame_id_list], output_folder=frames_folder)

        for i, img_i in enumerate(sorted(os.listdir(frames_folder))):
            body_pose_.append(smpl_poses[i][1:])
            global_orient_.append(smpl_poses[i][:1])
            smpl_betas_.append(smpl_betas[i])
            verts_.append(verts[i])
            pred_cams_.append(pred_cams[i])
            bboxes_xyxy_.append(bboxes_xyxy[i])
            image_path_.append(os.path.join('images', img_i))
            person_id_.append(0)
            frame_id_.append(frame_id_list[i])

        smpl = {}
        smpl['body_pose'] = np.array(body_pose_).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(global_orient_).reshape((-1, 3))
        smpl['betas'] = np.array(smpl_betas_).reshape((-1, 10))
        human_data['smpl'] = smpl
        human_data['verts'] = verts_
        human_data['pred_cams'] = pred_cams_
        human_data['bboxes_xyxy'] = bboxes_xyxy_
        human_data['image_path'] = image_path_
        human_data['person_id'] = person_id_
        human_data['frame_id'] = frame_id_
        human_data.dump(osp.join(args.output, 'inference_result.npz'))
 

    if args.show_path is not None:
        if args.output is not None:
            frames_folder = os.path.join(args.output, 'images')
        else:
            frames_folder = osp.join(Path(args.show_path).parent, 'images')
            os.makedirs(frames_folder, exist_ok=True)
            array_to_images(
                np.array(frames_iter)[frame_id_list],
                output_folder=frames_folder)

        body_model_config = dict(model_path=args.body_model_dir, type='smpl')
        visualize_smpl_hmr(
            poses=smpl_poses.reshape(-1, 24 * 3),
            betas=smpl_betas,
            cam_transl=pred_cams,
            bbox=bboxes_xyxy,
            output_path=args.show_path,
            render_choice=args.render_choice,
            resolution=frames_iter[0].shape[:2],
            origin_frames=frames_folder,
            body_model_config=body_model_config,
            overwrite=True,
            palette=args.palette,
            read_frames_batch=True)
        if args.output is None:
            shutil.rmtree(frames_folder)


def main(args):

    # prepare input
    

    if args.single_person_demo:
        single_person_with_mmdet(args, True)


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
