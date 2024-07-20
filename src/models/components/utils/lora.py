import torch
import torch.nn as nn


class _LORA_attention(nn.Module):
    def __init__(self, old_attention, r) -> None:
        super().__init__()
        self.att = old_attention
        self.dim = self.att.num_heads * self.att.head_dim
        self.in_layer_q = nn.Linear(self.dim, r)
        self.out_layer_q = nn.Linear(r, self.dim)
        self.in_layer_k = nn.Linear(self.dim, r)
        self.out_layer_k = nn.Linear(r, self.dim)
        self.in_layer_v = nn.Linear(self.dim, r)
        self.out_layer_v = nn.Linear(r, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.att.qkv(x)
        lora_qkv = torch.cat(
            [
                self.out_layer_q(self.in_layer_q(x)),
                self.out_layer_k(self.in_layer_k(x)),
                self.out_layer_v(self.in_layer_v(x)),
            ],
            dim=-1,
        )

        qkv = qkv.reshape(
            B, N, 3, self.att.num_heads, self.att.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.att.q_norm(q), self.att.k_norm(k)

        if self.att.fused_attn:
            x = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.att.attn_drop.p if self.att.training else 0.0,
            )
        else:
            q = q * self.att.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.att.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.att.proj(x)
        x = self.att.proj_drop(x)
        return x

    def freeze_not_lora(self):
        for param in self.att.parameters():
            param.requires_grad = False


def apply_lora(model, r):
    for i, block in enumerate(model.blocks):
        if hasattr(block, "attn"):
            model.blocks[i].attn = _LORA_attention(block.attn, r)
            model.blocks[i].attn.freeze_not_lora()
