import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def infer_output(model, num_channels, img_size, **kwargs):
    """Infer the output dimensions of the backbone model.

    Args:
        model (nn.Module): The backbone model.
        num_channels (int): Number of input channels.
        img_size (int): Size of the input image.

    Returns:
        embed_dim (int): The embedding dimension of the backbone model.
        downsample_factor (int): The downsample factor of the backbone model.
        feature_size (int): The size of the output feature map.
        features_format (str): The format of the output features.

    Raises:
        ValueError: If the backbone output features shape is not of dimension 3 or 4.
    """
    dumy_image = torch.rand(
        1, num_channels, img_size, img_size, dtype=torch.float32
    )
    with torch.no_grad():
        dumy_features = model.forward_features(dumy_image, **kwargs)

    def analyse(img_size, dumy_features):
        remove_cls_token = False

        print(f"backbone output features.shape: {dumy_features.shape}")
        if len(dumy_features.shape) == 4:
            embed_dim = dumy_features.shape[-3]
            downsample_factor = img_size // dumy_features.shape[-1]
            feature_size = dumy_features.shape[-1]
            features_format = "NCHW"
        elif len(dumy_features.shape) == 3:
            embed_dim = dumy_features.shape[-1]
            feature_size = math.floor(math.sqrt(dumy_features.shape[-2]))
            if feature_size**2 != dumy_features.shape[-2]:
                if feature_size**2 == dumy_features.shape[-2] - 1:
                    remove_cls_token = True
                else:
                    raise ValueError(
                        f"backbone output features.shape[-2] must be a square number? Currently it is {dumy_features.shape[-2]}"
                    )
            downsample_factor = img_size // feature_size
            features_format = "NLC"
        else:
            print(model)
            raise ValueError(
                f"backbone output features.shape must be of dimension 3 or 4? Currently it is {dumy_features.shape}"
            )
        return (
            embed_dim,
            downsample_factor,
            feature_size,
            features_format,
            remove_cls_token,
        )

    if isinstance(dumy_features, (list, tuple)):
        (
            embed_dim,
            downsample_factor,
            feature_size,
            features_format,
            remove_cls_token,
        ) = zip(*[analyse(img_size, f) for f in dumy_features])
        assert all(
            [cls == remove_cls_token[0] for cls in remove_cls_token]
        ), "All backbone output features must have the same remove_cls_token"
        assert all(
            [f == features_format[0] for f in features_format]
        ), "All backbone output features must have the same format"
        embed_dim = list(embed_dim)
        downsample_factor = list(downsample_factor)
        feature_size = list(feature_size)
        features_format = features_format[0]
        remove_cls_token = remove_cls_token[0]
    if isinstance(dumy_features, torch.Tensor):
        (
            embed_dim,
            downsample_factor,
            feature_size,
            features_format,
            remove_cls_token,
        ) = analyse(img_size, dumy_features)

    return (
        embed_dim,
        downsample_factor,
        feature_size,
        features_format,
        remove_cls_token,
    )


def load_state_dict(path, model_name="model"):
    """Load a state dict from a path.

    Args:
        path (str): The path to the state dict.
        model_name (str, optional): The name of the model. Defaults to "model".

    Returns:
        Dict[str, Any]: The state dict.
    """

    state_dict = torch.load(path, map_location="cpu")
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    # remove *.{model_name}. from the keys
    if model_name is not None:
        new_state_dict = {}
        for key in state_dict.keys():
            if model_name in key:
                new_key = re.sub(rf"^.*{model_name}\.", "", key)
                new_state_dict[new_key] = state_dict[key]

        return new_state_dict
    else:
        return state_dict


def patchify(x, p):
    """Patchify the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (N, L, C*(p**2)).
        p (int): Patch size.

    Returns:
        torch.Tensor: Patchified tensor of shape (N, h * w, p**2 * c).

    Raises:
        AssertionError: If the input tensor shape is not valid.
    """
    assert x.shape[2] == x.shape[3] and x.shape[2] % p == 0

    c = x.shape[1]
    h = w = x.shape[2] // p
    out = x.reshape(shape=(x.shape[0], c, h, p, w, p))
    out = torch.einsum("nchpwq->nhwcpq", out)
    out = out.reshape(shape=(x.shape[0], h * w, p**2 * c))
    return out


def unpatchify(x, p, c):
    """
    x: (N, L, C*patch_size**2)
    p: Patch embed patch size
    c: Num channels
    imgs: (N, C, H, W)
    """
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
    x = torch.einsum("nhwcpq->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs


def set_first_layer(model, n_channels, is_rgb=None):
    """Set the weight of the first layer of the model. using the following steps.

    -Replace the first layer 3->D with a layer (3+K)->D.
    -Copy the weights corresponding to 3->D from network RGB.
    -Initialize randomly the weights corresponding to K->D with weak values like N(0, 0.01), so that at the beginning it does not break everything.
    Args:
        previous_weight (torch.Tensor): The weight of the RGB first layer.
        n_channels (int): The number of channels of the new model.
        is_rgb (bool, optional): If the first 3 layers of the new model are RGB. Defaults to None.
    """
    if n_channels == 3:
        # this shouldn't be necessary but is a safety measure
        return

    if is_rgb is None:
        is_rgb = n_channels >= 3

    if is_rgb:
        assert (
            n_channels > 3
        ), "The number of channels must be greater than 3 if the new model use RGB"

    # get first conv or Linear
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            break
        if isinstance(module, nn.Linear):
            break
    previous_weight = module.weight.detach()

    # If the first layer is a convolutional layer
    if previous_weight.dim() == 4:
        # If the new model uses RGB
        if is_rgb:
            n_out = previous_weight.shape[0]
            assert (
                previous_weight.shape[1] == 3
            ), f"old weights must have 3 channels (RGB) found {previous_weight.shape[1]}"
            # Initialize randomly the weights corresponding to K->D with weak values like N(0, 0.01)
            new_weight = torch.randn(
                (
                    n_out,
                    n_channels,
                    previous_weight.shape[2],
                    previous_weight.shape[3],
                )
            )
            # Copy the weights corresponding to 3->D from network RGB
            new_weight[:, :3] = previous_weight
        else:
            # compute a mean value for the new weights
            mean = previous_weight.mean(dim=1)
            # Initialize the new weights with the mean value
            new_weight = torch.stack([mean] * n_channels, dim=1)

    # If the first layer is a linear layer (as with ViT)
    elif previous_weight.dim() == 2:
        n_out = previous_weight.shape[0]
        n_elem = previous_weight.shape[1] // 3
        # If the new model uses RGB
        if is_rgb:
            log.warning(
                "Converting a Linear layer. The RGB channels must be group together (if patch embedding (C p1 p2))"
            )
            # Initialize randomly the weights corresponding to K->D with weak values like N(0, 0.01)
            new_weight = torch.randn((n_out, n_channels * n_elem))
            # Copy the weights corresponding to 3->D from network RGB considering everything is flatenned
            new_weight[:, : 3 * n_elem] = previous_weight
            # new_weight[:, ::n_channels] = previous_weight[:, ::3]
            # new_weight[:, 1::n_channels] = previous_weight[:, 1::3]
            # new_weight[:, 2::n_channels] = previous_weight[:, 2::3]

        else:
            # compute a mean value for the new weights
            mean = previous_weight.reshape(n_out, -1, 3).mean(dim=-1)
            # Initialize the new weights with the mean value
            new_weight = torch.stack([mean] * n_channels, dim=-1)
            new_weight = new_weight.reshape(n_out, -1)

    module.weight = nn.parameter.Parameter(new_weight)
