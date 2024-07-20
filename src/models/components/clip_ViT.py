######################################
#
# From : https://github.com/openai/CLIP/blob/main/clip/model.py
#
######################################


import math
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.utils import (
    SimpleSegmentationHead,
    infer_output,
    load_state_dict,
    set_first_layer,
)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """A ResNet class that is similar to torchvision's but contains the following changes:

    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(
        self, layers, output_dim, heads, input_resolution=224, width=64
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = (
            width  # this is a *mutable* variable used during construction
        )
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask
        )[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(width, heads, attn_mask)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward_features(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(
            x.shape[0], x.shape[1], -1
        )  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class Clip_ViT(nn.Module):
    """HViT model implementation."""

    def __init__(
        self,
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

        self.ViT = VisionTransformer(
            input_resolution=img_size,
            patch_size=ViT_patch_size,
            output_dim=2,  # This value has no importance
            width=ViT_embed_dim,
            layers=ViT_depth,
            heads=ViT_num_heads,
        )

        # Load pretrained encoder
        if pretrained_encoder:
            assert (
                pretrained_encoder_path is not None
            ), "pretrained_encoder_path must be specified"
            if "laion" in pretrained_encoder_path:
                model_name = "visual"
            else:
                # OpenAI
                model_name = "vision_model"
            state_dict = load_state_dict(
                pretrained_encoder_path, model_name=model_name
            )

            # remove input/output dependent weight
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k not in ["head.weight", "head.bias", "proj"]
            }

            self.ViT.load_state_dict(state_dict, strict=False)
            set_first_layer(self.ViT, num_channels)

        if chkpt_path:
            raise NotImplementedError(
                "Checkpoint loading is not implemented yet"
            )

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
