import collections
from enum import Enum, auto

import torch
import torch.nn as nn
import torchvision

from src.models.components.utils.seg_blocks import SimpleSegmentationHead
from src.models.components.utils.utils import infer_output, set_first_layer


class Backbone(Enum):
    SWINB = auto()
    SWINT = auto()
    RESNET50 = auto()
    RESNET152 = auto()


class Head(Enum):
    CLASSIFY = auto()
    MULTICLASSIFY = auto()
    DETECT = auto()
    INSTANCE = auto()
    SEGMENT = auto()
    BINSEGMENT = auto()
    REGRESS = auto()


def adjust_state_dict_prefix(
    state_dict, needed, prefix=None, prefix_allowed_count=None
):
    """Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with
    'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component.
        if not needed in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, "", 1)

        new_state_dict[key] = value
    return new_state_dict


class SwinBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch):
        super().__init__()

        if arch == "swinb":
            self.backbone = torchvision.models.swin_v2_b()
            self.out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
        elif arch == "swint":
            self.backbone = torchvision.models.swin_v2_t()
            self.out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        else:
            raise ValueError("Backbone architecture not supported.")

        self.backbone.features[0][0] = torch.nn.Conv2d(
            num_channels,
            self.backbone.features[0][0].out_channels,
            kernel_size=(4, 4),
            stride=(4, 4),
        )

    def forward(self, x):
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class FPN(torch.nn.Module):
    def __init__(self, backbone_channels):
        super().__init__()

        out_channels = 128
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=out_channels
        )

        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x):
        inp = collections.OrderedDict(
            [(f"feat{i}", el) for i, el in enumerate(x)]
        )
        output = self.fpn(inp)
        output = list(output.values())

        return output


class Upsample(torch.nn.Module):
    # Computes an output feature map at 1x the input resolution.
    # It just applies a series of transpose convolution layers on the
    # highest resolution features from the backbone (FPN should be applied first).

    def __init__(self, backbone_channels):
        super().__init__()
        self.in_channels = backbone_channels

        out_channels = backbone_channels[0][1]
        self.out_channels = [(1, out_channels)] + backbone_channels

        layers = []
        depth, ch = backbone_channels[0]
        while depth > 1:
            next_ch = max(ch // 2, out_channels)
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(ch, ch, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)
            ch = next_ch
            depth /= 2

        self.layers = torch.nn.Sequential(*layers)


class Model(torch.nn.Module):
    def __init__(
        self,
        num_channels=3,
        multi_image=False,
        backbone=Backbone.SWINB,
        fpn=False,
        head=None,
        num_categories=None,
        weights=None,
    ):
        """Initializes a model, based on desired imagery source and model components. This class
        can be used directly to create a randomly initialized model (if weights=None) or can be
        called from the Weights class to initialize a SatlasPretrain pretrained foundation model.

        Args:
            num_channels (int): Number of input channels that the backbone model should expect.
            multi_image (bool): Whether or not the model should expect single-image or multi-image input.
            backbone (Backbone): The architecture of the pretrained backbone. All image sources support SwinTransformer.
            fpn (bool): Whether or not to feed imagery through the pretrained Feature Pyramid Network after the backbone.
            head (Head): If specified, a randomly initialized head will be included in the model.
            num_categories (int): If a Head is being returned as part of the model, must specify how many outputs are wanted.
            weights (torch weights): Weights to be loaded into the model. Defaults to None (random initialization) unless
                                    initialized using the Weights class.
        """
        super().__init__()

        # Validate user-provided arguments.
        if not isinstance(backbone, Backbone):
            raise ValueError("Invalid backbone.")
        if head and not isinstance(head, Head):
            raise ValueError("Invalid head.")
        if head and (num_categories is None):
            raise ValueError("Must specify num_categories if head is desired.")

        self.backbone = self._initialize_backbone(
            num_channels, backbone, multi_image, weights
        )

        if fpn:
            self.fpn = self._initialize_fpn(
                self.backbone.out_channels, weights
            )
            self.upsample = Upsample(self.fpn.out_channels)
        else:
            self.fpn = None

        if head:
            raise NotImplementedError("Head not implemented.")
        else:
            self.head = None

    def _initialize_backbone(
        self, num_channels, backbone_arch, multi_image, weights
    ):
        # Load backbone model according to specified architecture.
        if backbone_arch == Backbone.SWINB:
            backbone = SwinBackbone(num_channels, arch="swinb")
        elif backbone_arch == Backbone.SWINT:
            backbone = SwinBackbone(num_channels, arch="swint")
        elif backbone_arch == Backbone.RESNET50:
            raise NotImplementedError("ResNet50 not implemented.")
        elif backbone_arch == Backbone.RESNET152:
            raise NotImplementedError("ResNet152 not implemented.")
        else:
            raise ValueError("Unsupported backbone architecture.")

        # No support for multi image
        prefix_allowed_count = 1

        # Load pretrained weights into the initialized backbone if weights were specified.
        if weights is not None:
            state_dict = adjust_state_dict_prefix(
                weights, "backbone", "backbone.", prefix_allowed_count
            )
            backbone.load_state_dict(state_dict)

        return backbone

    def _initialize_fpn(self, backbone_channels, weights):
        fpn = FPN(backbone_channels)

        # Load pretrained weights into the initialized FPN if weights were specified.
        if weights is not None:
            state_dict = adjust_state_dict_prefix(
                weights, "fpn", "intermediates.0.", 0
            )
            fpn.load_state_dict(state_dict)
        return fpn

    def forward(self, imgs, targets=None):
        # Define forward pass
        x = self.backbone(imgs)
        if self.fpn:
            x = self.fpn(x)
            x = self.upsample(x)
        if self.head:
            x, loss = self.head(imgs, x, targets)
            return x, loss
        return x


class Satlas(nn.Module):
    def __init__(
        self,
        use_FPN=False,
        num_channels=3,
        num_classes=1,
        img_size=256,
        segmentation_head=SimpleSegmentationHead,
        pretrained=True,
        pretrained_path=None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.img_size = img_size
        assert pretrained, "Only pretrained models are supported"

        weights = torch.load(pretrained_path)
        self.model = Model(
            3,
            False,
            Backbone.SWINB,
            fpn=use_FPN,
            head=None,
            num_categories=None,
            weights=weights,
        )

        set_first_layer(
            self.model, num_channels
        )  # convert to correct number of features

        # if features_only is True, map the output to forward_features
        if not hasattr(self.model, "forward_features"):
            self.model.forward_features = self.model.forward

        # Measure downsample factor
        (
            self.embed_dim,
            self.downsample_factor,
            self.feature_size,
            self.features_format,
            self.remove_cls_token,
        ) = infer_output(self.model, self.num_channels, self.img_size)

        # Add segmentation head
        self.seg_head = segmentation_head(
            self.embed_dim,
            self.downsample_factor,
            self.remove_cls_token,
            self.features_format,
            self.feature_size,
            self.num_classes,
        )

    def forward(self, x, metas=None):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the output tensor.
        """
        x = self.model.forward_features(x)

        x = self.seg_head(x)
        return {"out": x}
