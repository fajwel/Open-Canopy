import torch
import torch.nn as nn


class _LORA_linear(nn.Module):
    def __init__(
        self, old_linear, r, multiplicity=1, freeze_not_lora=True
    ) -> None:
        super().__init__()
        self.multiplicity = multiplicity
        self.old_linear = old_linear

        self.in_layer = nn.ModuleList(
            [
                nn.Linear(self.old_linear.in_features, r)
                for i in range(multiplicity)
            ]
        )
        self.out_layer = nn.ModuleList(
            [
                nn.Linear(r, self.old_linear.out_features // multiplicity)
                for i in range(multiplicity)
            ]
        )
        if freeze_not_lora:
            self.freeze_not_lora()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_org = self.old_linear(x)
        x_lora = torch.cat(
            [
                self.out_layer[i](self.in_layer[i](x))
                for i in range(self.multiplicity)
            ],
            dim=-1,
        )
        return x_org + x_lora

    def freeze_not_lora(self):
        for param in self.old_linear.parameters():
            param.requires_grad = False


def apply_lora_linear(attention, r, freeze_not_lora=True):
    if hasattr(attention, "qkv"):
        attention.qkv = _LORA_linear(
            attention.qkv, r, 3, freeze_not_lora=freeze_not_lora
        )
    if hasattr(attention, "kv"):
        attention.kv = _LORA_linear(
            attention.kv, r, 2, freeze_not_lora=freeze_not_lora
        )
    if hasattr(attention, "q"):
        attention.q = _LORA_linear(
            attention.q, r, 1, freeze_not_lora=freeze_not_lora
        )
    if hasattr(attention, "k"):
        attention.k = _LORA_linear(
            attention.k, r, 1, freeze_not_lora=freeze_not_lora
        )
    if hasattr(attention, "v"):
        attention.v = _LORA_linear(
            attention.v, r, 1, freeze_not_lora=freeze_not_lora
        )


def apply_lora(model, r, assert_num_att=True):
    num_att = 0
    print(type(model))

    if isinstance(model, nn.ModuleDict):
        list = model.values()
    elif hasattr(model, "blocks"):
        list = model.blocks
    else:
        list = []
    for i, block in enumerate(list):
        print(type(block))
        if hasattr(block, "attn"):
            apply_lora_linear(model.blocks[i].attn, r)
            num_att += 1
        elif isinstance(block, nn.Module):
            num_att += apply_lora(block, r, assert_num_att=False)

    if assert_num_att:
        assert num_att > 0, "Model must have attention blocks to apply lora"
    return num_att
