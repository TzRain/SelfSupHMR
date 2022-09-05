from multiprocessing.spawn import import_main_path


import numpy as np
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_hmr

def demo_render(smpl_poses,smpl_betas,pred_cams,image_array=None,show_path=None,img_size=224,palette='segmentation',render_choice='hq'):

    batch_size = len(smpl_poses)
    bboxes_xyxy = []
    
    for _ in range(batch_size):
        bboxes_xyxy.append([0,0,img_size,img_size])
    
    bboxes_xyxy = np.array(bboxes_xyxy)

    body_model_config = dict(model_path='data/body_models/', type='smpl')

    tensor = visualize_smpl_hmr(
        poses=smpl_poses.reshape(-1, 24 * 3),
        betas=smpl_betas,
        cam_transl=pred_cams,
        bbox=bboxes_xyxy,
        render_choice=render_choice,
        resolution=img_size,
        # image_array = image_array,
        body_model_config=body_model_config,
        overwrite=True,
        palette=palette,
        read_frames_batch=True)
    
    return tensor