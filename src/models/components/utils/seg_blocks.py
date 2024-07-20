import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import infer_output

# FPN call infer output inside will the other expect it to be call outside. This should be uniform.


class SimpleSegmentationHead(nn.Module):
    """Simple segmentation head."""

    def __init__(
        self,
        embed_dim,
        downsample_factor,
        remove_cls_token,
        features_format,
        features_sizes,
        num_classes,
        decoder_stride=2,
        **kwargs,
    ):
        """Simple segmentation head.

        Args:
            embed_dim (int): Embedding dimension of the backbone model.
            downsample_factor (int): The downsample factor of the backbone model.
            remove_cls_token (bool): Whether to remove the cls token from the output features.
            features_format (str): The format of the output features.
            features_sizes (int): The size of the output feature map.
            num_classes (int): Number of classes.
            decoder_stride (int): The stride of the decoder.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.downsample_factor = downsample_factor
        self.remove_cls_token = remove_cls_token
        self.features_format = features_format
        self.feature_size = features_sizes
        self.num_classes = num_classes
        self.decoder_stride = decoder_stride

        self.layered_output = isinstance(self.embed_dim, (list, tuple))
        if self.layered_output:
            self.embed_dim = self.embed_dim[-1]
            self.downsample_factor = self.downsample_factor[-1]
            self.feature_size = self.feature_size[-1]
        print(
            f"{self.embed_dim=}, {self.downsample_factor=}, {self.feature_size=}"
        )
        depth = math.log(self.downsample_factor, decoder_stride)
        assert (
            depth.is_integer()
        ), f"decoder stride({decoder_stride}) must be a power of the downsample factor({self.downsample_factor})"
        depth = int(depth)
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.embed_dim // 2 ** (d),
                        self.embed_dim // 2 ** (d + 1),
                        decoder_stride,
                        stride=decoder_stride,
                    ),
                    nn.BatchNorm2d(self.embed_dim // 2 ** (d + 1)),
                    nn.GELU(),
                    nn.Conv2d(
                        self.embed_dim // 2 ** (d + 1),
                        self.embed_dim // 2 ** (d + 1),
                        3,
                        padding="same",
                    ),
                    nn.BatchNorm2d(self.embed_dim // 2 ** (d + 1)),
                    nn.GELU(),
                )
                for d in range(depth - 1)
            ]
            + [
                nn.ConvTranspose2d(
                    self.embed_dim // 2 ** (depth - 1),
                    num_classes,
                    decoder_stride,
                    stride=decoder_stride,
                )
            ]
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): The input to the segmentation head.

        Returns:
            torch.Tensor: The output of the segmentation head.
        """
        if self.layered_output:
            x = x[-1]
        if self.remove_cls_token:
            x = x[:, 1:, :]
        if self.features_format == "NLC":
            # Convert from NLC to NCHW
            x = x.reshape(
                x.shape[0], self.feature_size, self.feature_size, x.shape[-1]
            )
            x = x.permute(0, 3, 1, 2)
        return self.layers(x)


class ConvolutionnalEncoder(nn.Module):
    """Symmetrical opposite of the SimpleSegmentationHead."""

    def __init__(
        self,
        in_channels,
        downsample_factor,
        features_format,
        embed_dim,
        encoder_stride=2,
        **kwargs,
    ):
        """Simple segmentation head.

        Args:
            in_channels (int): Number of input channels.
            downsample_factor (int): The downsample factor of the backbone model.
            features_format (str): The format of the input features.
            embed_dim (int): Embedding dimension of the backbone model.
            decoder_stride (int): The stride of the decoder. Defaults to 2.
        """
        super().__init__()
        self.in_channels = in_channels
        self.downsample_factor = downsample_factor
        self.features_format = features_format
        self.embed_dim = embed_dim
        self.encoder_stride = encoder_stride

        depth = math.log(self.downsample_factor, encoder_stride)
        assert (
            depth.is_integer()
        ), f"decoder stride({encoder_stride}) must be a power of the downsample factor({self.downsample_factor})"
        depth = int(depth)
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        self.embed_dim // 2 ** (depth),
                        3,
                        padding="same",
                    ),
                    nn.BatchNorm2d(self.embed_dim // 2 ** (depth)),
                    nn.GELU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(
                        self.embed_dim // 2 ** (d),
                        self.embed_dim // 2 ** (d - 1),
                        encoder_stride,
                        stride=encoder_stride,
                    ),
                    nn.BatchNorm2d(self.embed_dim // 2 ** (d - 1)),
                    nn.GELU(),
                    nn.Conv2d(
                        self.embed_dim // 2 ** (d - 1),
                        self.embed_dim // 2 ** (d - 1),
                        3,
                        padding="same",
                    ),
                    nn.BatchNorm2d(self.embed_dim // 2 ** (d - 1)),
                    nn.GELU(),
                )
                for d in range(depth, 0, -1)
            ]
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): The input to the segmentation head.

        Returns:
            torch.Tensor: The output of the segmentation head.
        """
        if self.features_format == "NCHW":
            # Convert from NCHW to NLChw where hw are the downsample_factor
            x = x.view(
                x.shape[0],
                x.shape[1],
                x.shape[2] // self.downsample_factor,
                self.downsample_factor,
                x.shape[3] // self.downsample_factor,
                self.downsample_factor,
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(
                x.shape[0],
                x.shape[1] * x.shape[2],
                x.shape[3],
                x.shape[4],
                x.shape[5],
            )
        elif self.features_format == "NLCHW":
            pass
        else:
            raise NotImplementedError(
                f"ConvolutionnalEncoder not implemented for {self.features_format}"
            )
        N, L = x.shape[0], x.shape[1]
        x = x.reshape(
            N * L,
            x.shape[2],
            x.shape[3],
            x.shape[4],
        )
        out = self.layers(x).reshape(N, L, self.embed_dim)
        return out


class FPN(nn.Module):
    def __init__(self, model, num_channels, img_size):
        """Feature Pyramid Network intermediate module.

        Args:
            model (nn.Module): The backbone model.
            num_channels (int): Number of input channels.
            img_size (int): Size of the input image.
        """
        super().__init__()
        self.model = model
        self.num_channels = num_channels
        self.img_size = img_size

        (
            self.embed_dim,
            self.downsample_factor,
            self.feature_size,
            self.features_format,
            self.remove_cls_token,
        ) = infer_output(self.model, self.num_channels, self.img_size)
        assert isinstance(
            self.embed_dim, (list, tuple)
        ), "FPN requires layered output"

        self.convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(len(self.embed_dim)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(self.embed_dim[i], self.embed_dim[-1], 1),
                    nn.BatchNorm2d(self.embed_dim[-1]),
                    nn.GELU(),
                )
            )

    def forward_features(self, x):
        """Forward pass for extracting multiscale features.

        Args:
            x (nn.Tensor): The input to the backbone model.

        Returns:
            List[nn.Tensor]: The multiscale features.
        """
        x = self.model.forward_features(x)
        if self.remove_cls_token:
            x = [tensor[:, 1:, :] for tensor in x]
        if self.features_format == "NLC":
            # Convert from NLC to NCHW
            x = [
                tensor.reshape(
                    tensor.shape[0],
                    self.feature_size[i],
                    self.feature_size[i],
                    tensor.shape[-1],
                )
                for i, tensor in enumerate(x)
            ]
            x = [tensor.permute(0, 3, 1, 2) for tensor in x]

        out = []
        out.append(self.convs[-1](x[-1]))
        for i in range(len(x) - 2, -1, -1):
            out.append(
                self.convs[i](x[i])
                + nn.functional.interpolate(
                    out[-1],
                    size=x[i].shape[-2:],
                    align_corners=True,
                    mode="bilinear",
                )
            )

        return out[::-1]

    def forward(self, x):
        raise NotImplementedError(
            "FPN can only be used for feature extraction"
        )


class MultiscaleSegmentationHead(nn.Module):
    """Segmentation head inspire by semantic FPN."""

    def __init__(
        self,
        embed_dim,
        downsample_factor,
        remove_cls_token,
        features_format,
        features_sizes,
        num_classes,
        decoder_stride=2,
        fuse_method="sum",
        segmentation_head_width=1,
        **kwargs,
    ):
        """Segmentation head inspire by semantic FPN.

        Args:
            embed_dim (int): Embedding dimension of the backbone model.
            downsample_factor (int): The downsample factor of the backbone model.
            remove_cls_token (bool): Whether to remove the cls token from the output features.
            features_format (str): The format of the output features.
            features_sizes (int): The size of the output feature map.
            num_classes (int): Number of classes.
            decoder_stride (int, optional): The stride of the decoder. Defaults to 2.
            fuse_method (str, optional): The method used to fuse the features. Defaults to "sum".
            segmentation_head_width (int, optional): The width of the segmentation head (multiplied by num_classes). Defaults to 1.
        """
        super().__init__()
        self.fuse_method = fuse_method
        self.embed_dim = embed_dim
        self.downsample_factor = downsample_factor
        self.remove_cls_token = remove_cls_token
        self.features_format = features_format
        self.feature_size = features_sizes
        self.num_classes = num_classes
        self.decoder_stride = decoder_stride
        self.segmentation_head_width = segmentation_head_width

        if self.fuse_method == "sum":
            assert all(
                [
                    self.embed_dim[0] == embed_dim
                    for embed_dim in self.embed_dim
                ]
            ), "All layers must have the same embedding dimension"

        min_downsample_factor = min(self.downsample_factor)
        # create all up modules
        self.up = nn.ModuleList()
        for i in range(len(self.downsample_factor)):
            down_fact = self.downsample_factor[i] // min_downsample_factor
            # alternate between simple conv2d and bilinare upsample(x2)
            depth = math.log(down_fact, decoder_stride)
            assert (
                depth.is_integer()
            ), f"decoder stride({decoder_stride}) must be a power of the downsample factor({down_fact})"
            depth = int(depth)
            if depth == 0:
                self.up.append(
                    nn.Conv2d(
                        self.embed_dim[i], self.embed_dim[i], 3, padding="same"
                    )
                )
            else:
                up = nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv2d(
                                self.embed_dim[i],
                                self.embed_dim[i],
                                3,
                                padding="same",
                            ),
                            nn.BatchNorm2d(self.embed_dim[i]),
                            nn.GELU(),
                            nn.Upsample(
                                scale_factor=decoder_stride, mode="bilinear"
                            ),
                        )
                        for _ in range(depth)
                    ]
                )
                self.up.append(up)

        # Segmentation head
        if self.fuse_method == "sum":
            final_embed_dim = self.embed_dim[0]
        if self.fuse_method == "concat":
            final_embed_dim = sum(self.embed_dim)

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                final_embed_dim,
                num_classes * self.segmentation_head_width,
                3,
                padding="same",
            ),
            nn.BatchNorm2d(num_classes * self.segmentation_head_width),
            nn.GELU(),
            nn.Upsample(
                scale_factor=min_downsample_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.Conv2d(
                num_classes * self.segmentation_head_width,
                num_classes,
                3,
                padding="same",
            ),
        )

    def forward(self, x):
        # preprocess data to right format
        if self.remove_cls_token:
            x = [tensor[:, 1:, :] for tensor in x]
        if self.features_format == "NLC":
            # Convert from NLC to NCHW
            x = [
                tensor.reshape(
                    tensor.shape[0],
                    self.feature_size[i],
                    self.feature_size[i],
                    tensor.shape[-1],
                ).permute(0, 3, 1, 2)
                for i, tensor in enumerate(x)
            ]

        # Upsample and fuse
        x = [self.up[i](x[i]) for i in range(len(x))]
        if self.fuse_method == "sum":
            x = torch.stack(x, dim=0).sum(dim=0)
        if self.fuse_method == "concat":
            x = torch.cat(x, dim=1)
        x = self.final_conv(x)
        return x


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    This class is heavely inspired from:
    https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/mmseg/models/decode_heads/psp_head.py

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channels, self.channels, 1),
                    nn.BatchNorm2d(self.channels),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.
    <https://arxiv.org/abs/1807.10221>`

    This class is heavely inspired from:
    https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/mmseg/models/decode_heads/uper_head.py

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(
        self,
        embed_dim,
        downsample_factor,
        remove_cls_token,
        features_format,
        feature_size,
        num_classes,
        pool_scales=(1, 2, 3, 6),
        channels=512,
        align_corners=True,
        transform_inputs=nn.Identity(),
        **kwargs,
    ):
        super().__init__()
        self.transform_inputs = transform_inputs

        self.embed_dim = embed_dim
        self.downsample_factor = downsample_factor
        self.remove_cls_token = remove_cls_token
        self.features_format = features_format
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.channels = channels
        self.align_corners = align_corners

        assert isinstance(
            self.embed_dim, (list, tuple)
        ), "UperHead requires layered output"

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.embed_dim[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                self.embed_dim[-1] + len(pool_scales) * self.channels,
                self.channels,
                3,
            ),
            nn.BatchNorm2d(self.channels),
            nn.GELU(),
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in self.embed_dim[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channel, self.channels, 1),
                nn.BatchNorm2d(self.channels),
                nn.GELU(),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 3, padding=1),
                nn.BatchNorm2d(self.channels),
                nn.GELU(),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                len(self.embed_dim) * self.channels,
                self.channels,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(self.channels),
            nn.GELU(),
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.num_classes,
                3,
                padding=1,
            )
        )

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self.transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        fpn_outs = torch.cat(fpn_outs, dim=1)

        output = self.fpn_bottleneck(fpn_outs)
        output = F.interpolate(
            output,
            scale_factor=self.downsample_factor[0],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        output = self.cls_seg(output)
        return output
