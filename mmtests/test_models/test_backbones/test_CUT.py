import pytest
import torch
from mmcv import assert_params_all_zeros
from torch.nn.modules import AvgPool2d, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmhuman3d.models.backbones.builder import Encoder

def test_Encoder():
    model = Encoder(batch_size=8)
    print(model)

if __name__=='__main__':
    test_Encoder()
    