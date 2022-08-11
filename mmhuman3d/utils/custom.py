

import cv2
import torch
import numpy as np
import mmhuman3d.core.visualization.visualize_smpl as visualize_smpl

from datetime import datetime
from mmhuman3d.utils.transforms import rotmat_to_aa

def custom_renderer(results,save_image=True):
    smpl_poses = results['pred_pose']
    smpl_betas = results['pred_betas']
    pred_cams = results['pred_cam']
    affined_img = results['affined_img']

    print("run custom renderer")

    # smpl_poses = np.array(smpl_poses)
    # smpl_betas = np.array(smpl_betas)
    # pred_cams = np.array(pred_cams)
    # affined_img = np.array(affined_img)

    if smpl_poses.shape[1:] == (24, 3, 3):
        smpl_poses = rotmat_to_aa(smpl_poses)

    body_model_config = dict(model_path="data/body_models/", type='smpl')
    tensors = visualize_smpl.visualize_smpl_hmr(
        poses=smpl_poses.reshape(-1, 24 * 3),
        betas=smpl_betas,
        cam_transl=pred_cams,
        render_choice='hq',
        resolution=affined_img[0].shape[:2],
        image_array=affined_img,
        body_model_config=body_model_config,
        return_tensor = True,
        no_grad = False,
        palette='segmentation',
        read_frames_batch=False,
        batch_size = affined_img[0].shape[0],
    )
    tensors_de = tensors.detach()

    if save_image == True:
        save_img(affined_img,path_folders='affined_image')
        save_img(tensors_de,path_folders='rendered_image')

    return tensors



def save_img(img,path_folders='affined_image',title=None):
    if title is None:
        now = datetime.now()
        title = now.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(img,list):
        for i,im in enumerate(img):
            print(f"write vis_results/{path_folders}/{title}-{i}.jpg")
            cv2.imwrite(f"vis_results/{path_folders}/{title}-{i}.jpg",im)
    else:
        if isinstance(img,torch.Tensor):
            img = img.cpu().numpy()

        if img.ndim == 3:    
            print(f"write vis_results/{path_folders}/{title}.jpg")
            cv2.imwrite(f"vis_results/{path_folders}/{title}.jpg",img)
        
        elif img.ndim == 4:    
            for i in range(img.shape[0]):
                im  = img[i]
                print(f"write vis_results/{path_folders}/{title}-{i}.jpg")
                cv2.imwrite(f"vis_results/{path_folders}/{title}-{i}.jpg",im)
        else :
            print("unexcepted img ndim")