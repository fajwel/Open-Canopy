import pytest
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from numpy import imag

from src.models.components.utils import set_first_layer


def test_set_first_layer():
    image = torch.randn(1, 3, 16, 16)
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
    )
    out_1 = model(image)

    set_first_layer(model, 5, is_rgb=True)
    image_extended = torch.zeros(1, 5, 16, 16)
    image_extended[:, :3] = image
    out_2 = model(image_extended)
    assert torch.allclose(
        out_1, out_2
    ), f"Conv2d not properly converted. Error {torch.abs(out_1 - out_2).max()}"

    model = nn.Sequential(nn.Linear(3 * 16 * 16, 64), nn.ReLU())
    rea = Rearrange(
        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
        p1=16,
        p2=16,
    )
    out_1 = model(rea(image))
    set_first_layer(model, 5, is_rgb=True)
    out_2 = model(rea(image_extended))
    assert torch.allclose(
        out_1, out_2
    ), f"Linear not properly converted. Error {torch.abs(out_1 - out_2).max()}"
