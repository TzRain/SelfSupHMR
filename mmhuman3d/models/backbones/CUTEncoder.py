# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from datetime import datetime


from CUT.models import networks
from CUT.models.patchnce import PatchNCELoss
from CUT.options.train_options import TrainOptions


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
        
        

    
    