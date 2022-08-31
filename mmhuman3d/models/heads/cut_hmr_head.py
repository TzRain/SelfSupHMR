import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from mmcv.runner.base_module import BaseModule
from mmhuman3d.utils.geometry import rot6d_to_rotmat
from CUT.models import networks
from CUT.models.patchnce import PatchNCELoss
from CUT.options.train_options import TrainOptions
from mmhuman3d.utils.custom import custom_renderer
class CUTHMRHead(BaseModule):
    def __init__(self,
                 feat_dim,
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
        self.save_image_time = datetime.now()
    
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
                img_metas = None,
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

        scale = [item['scale'] for item in img_metas]
        center = [item['center'] for item in img_metas]
        affined_img = torch.Tensor([item['affined_img'] for item in img_metas]).to(predictions['pred_pose'].device)  

        result = {}
        result['pred_pose'] = predictions['pred_pose']
        result['pred_betas'] = predictions['pred_shape']
        result['pred_cam'] = predictions['pred_cam']
        result['affined_img'] = affined_img
        result['scale'] = scale
        result['center'] = center

        now_time = datetime.now()
        save_image = False

        if (now_time - self.save_image_time).total_seconds() > 10 * 60:
            save_image = True
            self.save_image_time = now_time

        tensors =  custom_renderer(result,save_image=save_image)
        NCE_loss = self.calculate_NCE_loss(tensors,affined_img)
        return predictions , NCE_loss

if __name__ == "__main__":
    model = CUTHMRHead(feat_dim=2048)