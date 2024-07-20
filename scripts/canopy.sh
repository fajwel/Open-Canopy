#!/bin/bash

python src/train.py --multirun model=smp_unet model.loss=l1,l2,Huber
python src/train.py --multirun model=smp_unet model.activation=none,relu,softplus

python src/train.py --multirun model=smp_unet,DLv3
python src/train.py --multirun model=ViT_B,HViT_B,ViT_dino_B,ViT_clipAI_B
python src/train.py --multirun model=ViT_tolan_L
python src/train.py --multirun model=ScaleViT_L_pre
python src/train.py --multirun model=PCPVT_B,PVTv2_B,HViT_B
