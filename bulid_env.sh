#!/bin/bash
echo build env
# conda create -n SSHMR python=3.8 -y
# conda init
# conda activate SSHMR
conda install ffmpeg -y
conda install pytorch==1.8.0 torchvision cudatoolkit=10.2 -c pytorch -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
pip install "mmcv-full>=1.3.17,<1.6.0" -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
pip install "mmdet<=2.25.1"
pip install "mmpose<=0.28.1"
pip install "mmcls<=0.23.2" "mmtrack<=0.13.0"