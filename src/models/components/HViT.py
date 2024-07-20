import math
import stat
from functools import partial
from turtle import pos, st

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed as LinearEmbedding

from models.components.utils import (
    ConvolutionnalEncoder,
    SimpleSegmentationHead,
    infer_output,
    load_state_dict,
    set_first_layer,
)
from src.models.components.utils.lora import apply_lora


class SimpleConvBackbone:
    """Convolutionnal backbone for patch embedding with Conv2D, batch normalization, GELU
    activation, and max pooling."""

    pass  # declared later


class HViT(nn.Module):
    """HViT model implementation."""

    def __init__(
        self,
        embedder=ConvolutionnalEncoder,
        img_size=512,
        num_channels=1,
        num_classes=4,
        ViT_patch_size=16,
        ViT_embed_dim=768,
        ViT_depth=12,
        ViT_num_heads=12,
        ViT_mlp_ratio=4.0,
        chkpt_path=None,
        pretrained_encoder=False,
        pretrained_encoder_path=None,
        pretrained_encoder_inter=False,
        decoder_stride=16,
        lora_rank=0,
        model_filter_name=None,
    ):
        """HViT model implementation.

        Args:
            embedder (nn.Module, optional): Backbone network for feature extraction. Defaults to SimpleConvBackbone.
            img_size (int, optional): Size of the input image. Defaults to 512.
            num_channels (int, optional): Number of channels in the input image. Defaults to 1.
            num_classes (int, optional): Number of output classes. Defaults to 4.
            ViT_patch_size (int, optional): Patch size for the Vision Transformer. Defaults to 16.
            ViT_embed_dim (int, optional): Embedding dimension for the Vision Transformer. Defaults to 768.
            ViT_depth (int, optional): Depth of the Vision Transformer. Defaults to 12.
            ViT_num_heads (int, optional): Number of attention heads in the Vision Transformer. Defaults to 12.
            ViT_mlp_ratio (float, optional): Ratio of MLP hidden size to embedding dimension in the Vision Transformer. Defaults to 4.0.
        """
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_channels = num_channels

        class_tocken = "dino" in pretrained_encoder_path

        self.ViT = timm.models.VisionTransformer(
            img_size=img_size,
            patch_size=ViT_patch_size,
            in_chans=3,
            num_classes=2,  # This value has no importance
            global_pool="avg",  # This value has no importance
            embed_dim=ViT_embed_dim,
            depth=ViT_depth,
            num_heads=ViT_num_heads,
            mlp_ratio=ViT_mlp_ratio,
            class_token=class_tocken,
            embed_layer=embedder,
        )

        # Load pretrained encoder
        if pretrained_encoder:
            assert (
                pretrained_encoder_path is not None
            ), "pretrained_encoder_path must be specified"
            state_dict = load_state_dict(
                pretrained_encoder_path, model_name=model_filter_name
            )

            pos_embed = state_dict["pos_embed"]
            # remove input/output dependent weight
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k
                not in [
                    "pos_embed",
                    "head.weight",
                    "head.bias",
                ]
            }

            if pretrained_encoder_inter:
                # reshape pos_embed to 2D
                if class_tocken:
                    pos_embed = pos_embed[:, 1:]
                    cls_tocken = pos_embed[:, 0]
                size = int(math.sqrt(pos_embed.shape[1]))
                pos_embed = pos_embed.reshape(size, size, -1)
                # convert to CHW
                pos_embed = pos_embed.permute(2, 0, 1)
                # interpolate position embedding
                pos_embed = F.interpolate(
                    pos_embed.unsqueeze(0),
                    size=self.ViT.patch_embed.grid_size,
                    mode="bicubic",
                    align_corners=True,
                ).squeeze(0)
                # convert back to HWC
                pos_embed = pos_embed.permute(1, 2, 0)
                pos_embed = pos_embed.reshape(1, -1, ViT_embed_dim)
                if class_tocken:
                    state_dict["pos_embed"] = torch.cat(
                        (cls_tocken.unsqueeze(0), pos_embed), dim=1
                    )
                else:
                    state_dict["pos_embed"] = pos_embed

            self.ViT.load_state_dict(state_dict, strict=False)
            set_first_layer(self.ViT, num_channels)

        if chkpt_path:
            raise NotImplementedError(
                "Checkpoint loading is not implemented yet"
            )

        if lora_rank > 0:
            # freeze model
            for param in self.ViT.parameters():
                param.requires_grad = False
            apply_lora(self.ViT, lora_rank)

        # Measure downsample factor
        (
            self.embed_dim,
            self.downsample_factor,
            self.feature_size,
            self.features_format,
            self.remove_cls_token,
        ) = infer_output(self.ViT, num_channels, self.img_size)

        # Add segmentation head
        self.seg_head = SimpleSegmentationHead(
            self.embed_dim,
            self.downsample_factor,
            self.remove_cls_token,
            self.features_format,
            self.feature_size,
            self.num_classes,
            decoder_stride,
        )

    def forward(self, x, metas=None):
        """Forward pass of the HViT model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_chans, img_size, img_size).

        Returns:
            dict: A dictionary containing the output tensor of shape (batch_size, in_chans, img_size, img_size).
        """
        embedding = self.ViT.forward_features(x)
        # apply decoder
        x = self.seg_head(embedding)

        # if output size is bigger than input size, crop it
        assert (
            x.shape[-1] >= self.img_size
        ), "output size is smaller than input size, this is unexpected"
        if x.shape[-1] > self.img_size:
            delta = x.shape[-1] - self.img_size
            cropL = delta // 2
            cropR = delta - cropL
            x = x[:, :, cropL:-cropR, cropL:-cropR]
        return {"out": x}
