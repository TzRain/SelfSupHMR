# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from datetime import datetime


import networks


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool # torch version > '1.2.0'

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class Encoder(BaseModule):
    """

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the G'architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        netF (str) -- the F'architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """

    def __init__(self,
                batch_size,
                input_nc=3,
                output_nc=3,
                ngf=64,
                netG='resnet_9blocks',
                netF='mlp_sample',
                nce_layers='0,4,8,12,16',
                gpu_ids=[],
                norm='batch',
                init_type='normal',
                init_gain=0.02,
                use_dropout=False, 
                no_antialias=False,
                no_antialias_up=False,
                netF_nc=256,
                nce_T=0.07,
                nce_includes_all_negatives_from_minibatch=False,
                image_resolution=224,
                init_cfg=None):
        super(Encoder, self).__init__(init_cfg)

        class OPT:
            pass

        opt = OPT()
        param = ['batch_size','nce_T','netF_nc','nce_includes_all_negatives_from_minibatch']
        value = [batch_size,nce_T,netF_nc,nce_includes_all_negatives_from_minibatch]
        for p,v in zip(param,value):
            setattr(opt,p,v)
        
        self.opt = opt

        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.netG = networks.define_G(input_nc, output_nc, ngf, netG, norm, use_dropout, init_type, init_gain, no_antialias, no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(input_nc, netF, norm, use_dropout, init_type, init_gain, no_antialias, self.gpu_ids, opt)
        self.nce_layers = [int(i) for i in nce_layers.split(',')]
        self.num_patches = len(self.nce_layers)
        self.criterionNCE = []

        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
        self.save_image_time = datetime.now()
        # init 
        self.data_dependent_initialize(image_resolution)


    def forward(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.num_patches, sample_ids)
        total_nce_loss = 0.0
        
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def data_dependent_initialize(self, image_resolution=224):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        fake = torch.zeros(self.opt.batch_size,3,image_resolution,image_resolution)
        self.forward(fake,fake)
        
        

    
    