
import torch
import torch.nn as nn
import cv2
import numpy as np

from datetime import datetime
from mmcv.runner.base_module import BaseModule
from mmhuman3d.core.visualization import visualize_smpl
from mmhuman3d.utils.geometry import rot6d_to_rotmat
from mmhuman3d.utils.transforms import rotmat_to_aa

from CUT.models import networks
from CUT.models.patchnce import PatchNCELoss
from CUT.options.train_options import TrainOptions

def save_img(img,path_folders='affined_image',title=None):
    # not need it any more
    # return 
    if title is None:
        now = datetime.now()
        title = now.strftime("%H:%M:%S")
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

def custom_renderer(predictions,affined_img):
    smpl_poses = predictions['pred_pose']
    smpl_betas = predictions['pred_shape']
    pred_cams = predictions['pred_cam']
    affined_imgs = np.array(affined_img)

    print("run custom renderer")

    if smpl_poses.shape[1:] == (24, 3, 3):
        smpl_poses = rotmat_to_aa(smpl_poses)
    

    body_model_config = dict(model_path="data/body_models/", type='smpl')
    
    tensors = visualize_smpl.visualize_smpl_hmr(
        poses=smpl_poses.reshape(-1, 24 * 3),
        betas=smpl_betas,
        cam_transl=pred_cams,
        render_choice='hq',
        resolution=affined_imgs[0].shape[:2],
        image_array=affined_imgs,
        body_model_config=body_model_config,
        return_tensor = True,
        no_grad = False,
        palette='segmentation',
        read_frames_batch=True)

    return tensors

class CUTHMRHead(BaseModule):
    def __init__(self,
                 feat_dim,
                #  opt=None, #netG opt
                 smpl_mean_params=None,
                 npose=144,
                 nbeta=10,
                 ncam=3,
                 hdim=1024,
                 init_cfg=None):
        super(CUTHMRHead, self).__init__(init_cfg=init_cfg)
        self.fc1 = nn.Linear(feat_dim + npose + nbeta + ncam, hdim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hdim, hdim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hdim, npose)
        self.decshape = nn.Linear(hdim, nbeta)
        self.deccam = nn.Linear(hdim, ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        if smpl_mean_params is None:
            init_pose = torch.zeros([1, npose])
            init_shape = torch.zeros([1, nbeta])
            init_cam = torch.FloatTensor([[1, 0, 0]])
        else:
            mean_params = np.load(smpl_mean_params)
            init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
            init_shape = torch.from_numpy(
                mean_params['shape'][:].astype('float32')).unsqueeze(0)
            init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)

        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        self.opt = TrainOptions().parse()
        opt = self.opt

        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        
        self.nce_layers = [int(i) for i in opt.nce_layers.split(',')]
        
        self.criterionNCE = []
        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
    
    def calculate_NCE_loss(self, src, tgt): #[B,3,256,256] [B,3,256,256]
        # return 0
        src = src.permute(0,3,1,2)
        tgt = tgt.permute(0,3,1,2)

        n_layers = len(self.nce_layers)

        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)#! encode_only
        feat_k = self.netG(src, self.nce_layers, encode_only=True)#! encode_only
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers    

    def forward(self,
                x,
                init_pose=None,
                init_shape=None,
                init_cam=None,
                is_training = False,
                affined_imgs = None,
                n_iter=3):

        # hmr head only support one layer feature
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[-1]

        output_seq = False
        if len(x.shape) == 4:
            # use feature from the last layer of the backbone
            # apply global average pooling on the feature map
            x = x.mean(dim=-1).mean(dim=-1)
        elif len(x.shape) == 3:
            # temporal feature
            output_seq = True
            B, T, L = x.shape
            x = x.view(-1, L)

        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        if output_seq:
            pred_rotmat = pred_rotmat.view(B, T, 24, 3, 3)
            pred_shape = pred_shape.view(B, T, 10)
            pred_cam = pred_cam.view(B, T, 3)
        predictions = {
            'pred_pose': pred_rotmat,
            'pred_shape': pred_shape,
            'pred_cam': pred_cam
        }

        if not is_training:
            return predictions
        
        render_tensor =  custom_renderer(predictions,affined_imgs)

        render_tensor_de = render_tensor.detach()
        
        save_img(render_tensor_de,"rendered_image")

        # NCE_loss = self.calculate_NCE_loss(tensors,torch.Tensor(affined_img).to(tensors.device))
        NCE_loss = None
        
        return predictions , NCE_loss




# import numpy as np
# import torch
# import torch.nn as nn
# from mmcv.runner.base_module import BaseModule

# from mmhuman3d.utils.geometry import rot6d_to_rotmat


# class CUTHMRHead(BaseModule):

#     def __init__(self,
#                  feat_dim,
#                  smpl_mean_params=None,
#                  npose=144,
#                  nbeta=10,
#                  ncam=3,
#                  hdim=1024,
#                  init_cfg=None):
#         super(CUTHMRHead, self).__init__(init_cfg=init_cfg)
#         self.fc1 = nn.Linear(feat_dim + npose + nbeta + ncam, hdim)
#         self.drop1 = nn.Dropout()
#         self.fc2 = nn.Linear(hdim, hdim)
#         self.drop2 = nn.Dropout()
#         self.decpose = nn.Linear(hdim, npose)
#         self.decshape = nn.Linear(hdim, nbeta)
#         self.deccam = nn.Linear(hdim, ncam)

#         nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
#         nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
#         nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

#         if smpl_mean_params is None:
#             init_pose = torch.zeros([1, npose])
#             init_shape = torch.zeros([1, nbeta])
#             init_cam = torch.FloatTensor([[1, 0, 0]])
#         else:
#             mean_params = np.load(smpl_mean_params)
#             init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
#             init_shape = torch.from_numpy(
#                 mean_params['shape'][:].astype('float32')).unsqueeze(0)
#             init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
#         self.register_buffer('init_pose', init_pose)
#         self.register_buffer('init_shape', init_shape)
#         self.register_buffer('init_cam', init_cam)

#     def forward(self,
#                 x,
#                 init_pose=None,
#                 init_shape=None,
#                 init_cam=None,
#                 n_iter=3):

#         # hmr head only support one layer feature
#         if isinstance(x, list) or isinstance(x, tuple):
#             x = x[-1]

#         output_seq = False
#         if len(x.shape) == 4:
#             # use feature from the last layer of the backbone
#             # apply global average pooling on the feature map
#             x = x.mean(dim=-1).mean(dim=-1)
#         elif len(x.shape) == 3:
#             # temporal feature
#             output_seq = True
#             B, T, L = x.shape
#             x = x.view(-1, L)

#         batch_size = x.shape[0]
#         if init_pose is None:
#             init_pose = self.init_pose.expand(batch_size, -1)
#         if init_shape is None:
#             init_shape = self.init_shape.expand(batch_size, -1)
#         if init_cam is None:
#             init_cam = self.init_cam.expand(batch_size, -1)

#         pred_pose = init_pose
#         pred_shape = init_shape
#         pred_cam = init_cam
#         for i in range(n_iter):
#             xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
#             xc = self.fc1(xc)
#             xc = self.drop1(xc)
#             xc = self.fc2(xc)
#             xc = self.drop2(xc)
#             pred_pose = self.decpose(xc) + pred_pose
#             pred_shape = self.decshape(xc) + pred_shape
#             pred_cam = self.deccam(xc) + pred_cam

#         pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

#         if output_seq:
#             pred_rotmat = pred_rotmat.view(B, T, 24, 3, 3)
#             pred_shape = pred_shape.view(B, T, 10)
#             pred_cam = pred_cam.view(B, T, 3)
#         output = {
#             'pred_pose': pred_rotmat,
#             'pred_shape': pred_shape,
#             'pred_cam': pred_cam
#         }
#         return output


if __name__ == "__main__":
    model = CUTHMRHead(feat_dim=2048)