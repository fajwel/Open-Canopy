#!/bin/bash

cd datasets || exit
mkdir init_models
cd init_models || exit

curl -L https://download.pytorch.org/models/resnet34-333f7ec4.pth -o resnet34-333f7ec4.pth
curl -L https://download.pytorch.org/models/resnet50-11ad3fa6.pth -o resnet50-11ad3fa6.pth
curl -L https://huggingface.co/timm/vit_base_r50_s16_224.orig_in21k/resolve/main/pytorch_model.bin?download=true -o hvit_base_r50_s16_224.orig_in21k.bin
#curl -L https://huggingface.co/timm/vit_small_r26_s32_224.augreg_in21k/resolve/main/pytorch_model.bin?download=true -o hvit_small_r26_s32_224.augreg_in21k.bin
curl -L https://huggingface.co/timm/twins_pcpvt_base.in1k/resolve/main/pytorch_model.bin?download=true -o pcpvt_base_in1k.bin
#curl -L https://huggingface.co/timm/twins_pcpvt_small.in1k/resolve/main/pytorch_model.bin?download=true -o pcpvt_small_in1k.bin
#curl -L https://huggingface.co/Xrenya/pvt-small-224/resolve/main/pytorch_model.bin?download=true -o pvt_small.pth
curl -L https://huggingface.co/timm/pvt_v2_b3.in1k/resolve/main/pytorch_model.bin?download=true -o pvt_v2_b3.in1k.bin
#curl -L https://huggingface.co/timm/pvt_v2_b1.in1k/resolve/main/pytorch_model.bin?download=true -o pvt_v2_b1.in1k.bin
curl -L https://zenodo.org/records/7338613/files/pretrain-vit-base-e199.pth?download=1 -o satmae-vit-base-e199.pth
curl -L https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth -o scalemae-vitlarge-800.pth
curl -L https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k/resolve/main/pytorch_model.bin?download=true -o swin_base_patch4_window7_224.ms_in22k.bin
#curl -L https://huggingface.co/timm/swin_small_patch4_window7_224.ms_in22k/resolve/main/pytorch_model.bin?download=true -o swin_small_patch4_window7_224.ms_in22k.bin
curl -L https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth -o deit_3_base_224_21k.pth
#curl -L https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_21k.pth -o deit_3_medium_224_21k.pth
#curl -L https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth -o deit_3_small_224_21k.pth
curl -L https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -o dinov2_vitb14_pretrain.pth
#curl -L https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -o dinov2_vits14_pretrain.pth
curl -L https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/pytorch_model.bin?download=true -o clip-vit-base-patch16.bin
curl -L https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K/resolve/main/open_clip_pytorch_model.bin?download=true -o CLIP-ViT-B-16-laion2B-s34B-b88K.bin
curl -L  https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_si.pth?download=true -o SatLas_aerial_swinb_si.pth
