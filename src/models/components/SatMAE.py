####################################################################################################
# See https://github.com/stanfordmlgroup/USat/blob/main/usat/models/SatMAE.py
# And https://github.com/stanfordmlgroup/USat/blob/main/usat/models/SatViT.py
####################################################################################################
from functools import partial

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from src.models.components.utils import (
    SimpleSegmentationHead,
    infer_output,
    load_state_dict,
    patchify,
    unpatchify,
)
from src.models.components.utils.pos_emb import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class SatVitEncoderRGB(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling."""

    def __init__(self, global_pool=False, channel_groups=[], **kwargs):
        super().__init__(**kwargs)

        self.channel_groups = channel_groups

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, kwargs["embed_dim"]),
            requires_grad=False,
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs):
        x_c = []
        for i, group in enumerate(self.channel_groups):
            for band in group:
                x_c.append(imgs[band])
        x = torch.cat(x_c, dim=1)

        x = self.forward_features(x)
        return x


class SatViTEncoder(nn.Module):
    """Vision Transformer with support for global average pooling."""

    def __init__(
        self,
        channel_embed=256,
        img_size=224,
        patch_size=8,
        in_c=10,
        embed_dim=1024,
        channel_groups=None,
        num_heads=16,
        depth=24,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()

        self.channel_groups = channel_groups
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = nn.ModuleList(
            [
                PatchEmbed(img_size, patch_size, len(group), embed_dim)
                for group in channel_groups
            ]
        )
        # self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed[0].num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional and channel embed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim - channel_embed),
            requires_grad=False,
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        num_groups = len(channel_groups)
        self.channel_embed = nn.Parameter(
            torch.zeros(1, num_groups, channel_embed), requires_grad=False
        )
        chan_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], torch.arange(num_groups).numpy()
        )
        self.channel_embed.data.copy_(
            torch.from_numpy(chan_embed).float().unsqueeze(0)
        )

        # self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # Extra embedding for cls to fill embed_dim
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
        channel_cls_embed = torch.zeros((1, channel_embed))
        self.channel_cls_embed.data.copy_(
            channel_cls_embed.float().unsqueeze(0)
        )

    def forward_encoder(self, x):
        b, c, h, w = x.shape

        x_c_embed = []
        current = 0
        for i, group in enumerate(self.channel_groups):
            # The order of each group is just the order of x
            interval = torch.arange(current, current + len(group))
            current += len(group)
            x_c_embed.append(
                self.patch_embed[i](x[:, interval, :, :])
            )  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(
            -1, -1, pos_embed.shape[2], -1
        )  # (1, c, L, cD)
        pos_embed = pos_embed.expand(
            -1, channel_embed.shape[1], -1, -1
        )  # (1, c, L, pD)
        pos_channel = torch.cat(
            (pos_embed, channel_embed), dim=-1
        )  # (1, c, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)
        x = x.view(b, -1, D)  # (N, G*L, D)

        cls_pos_channel = torch.cat(
            (self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1
        )  # (1, 1, D)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D)
        # x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        outcome = self.norm(x)

        return outcome

    def forward_features(self, x):
        x = self.forward_encoder(x)
        return x

    def forward(self, imgs):
        x_c = []
        for i, group in enumerate(self.channel_groups):
            for band in group:
                x_c.append(imgs[band])
        x = torch.cat(x_c, dim=1)

        x = self.forward_encoder(x)
        return x


class SatMAE(SatViTEncoder):
    """Masked Autoencoder with VisionTransformer backbone."""

    def __init__(
        self,
        img_size=224,
        patch_size=8,
        in_chans=10,
        spatial_mask=False,
        channel_groups=None,
        # channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
        channel_embed=256,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_channel_embed=128,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        # THis inits the encoder specific details
        super().__init__(
            channel_embed=channel_embed,
            img_size=img_size,
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            channel_groups=channel_groups,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        self.in_c = in_chans
        self.patch_size = patch_size
        self.channel_groups = channel_groups
        self.spatial_mask = spatial_mask  # Whether to mask all channels of same spatial location
        num_groups = len(channel_groups)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        num_patches = self.patch_embed[0].num_patches
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, num_patches + 1, decoder_embed_dim - decoder_channel_embed
            ),
            requires_grad=False,
        )  # fixed sin-cos embedding
        # Extra channel for decoder to represent special place for cls token
        self.decoder_channel_embed = nn.Parameter(
            torch.zeros(1, num_groups + 1, decoder_channel_embed),
            requires_grad=False,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.ModuleList(
            [
                nn.Linear(decoder_embed_dim, len(group) * patch_size**2)
                for group in channel_groups
            ]
        )
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed[0].num_patches ** 0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1],
            torch.arange(len(self.channel_groups)).numpy(),
        )
        self.channel_embed.data.copy_(
            torch.from_numpy(channel_embed).float().unsqueeze(0)
        )

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed[0].num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        dec_channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_channel_embed.shape[-1],
            torch.arange(len(self.channel_groups) + 1).numpy(),
        )
        self.decoder_channel_embed.data.copy_(
            torch.from_numpy(dec_channel_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        for patch_embed in self.patch_embed:
            w = patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """Perform per-sample random masking by per-sample shuffling.

        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x is (N, C, H, W)
        b, c, h, w = x.shape

        x_c_embed = []
        current = 0
        for i, group in enumerate(self.channel_groups):
            # The order of each group is just the order of x
            interval = torch.arange(current, current + len(group))
            current += len(group)
            x_c_embed.append(
                self.patch_embed[i](x[:, interval, :, :])
            )  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, G, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(
            -1, -1, pos_embed.shape[2], -1
        )  # (1, G, L, cD)
        pos_embed = pos_embed.expand(
            -1, channel_embed.shape[1], -1, -1
        )  # (1, G, L, pD)
        pos_channel = torch.cat(
            (pos_embed, channel_embed), dim=-1
        )  # (1, G, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D)

        if self.spatial_mask:
            # Mask spatial location across all channels (i.e. spatial location as either all/no channels)
            x = x.permute(0, 2, 1, 3).reshape(b, L, -1)  # (N, L, G*D)
            x, mask, ids_restore = self.random_masking(
                x, mask_ratio
            )  # (N, 0.25*L, G*D)
            x = (
                x.view(b, x.shape[1], G, D)
                .permute(0, 2, 1, 3)
                .reshape(b, -1, D)
            )  # (N, 0.25*G*L, D)
            mask = mask.repeat(1, G)  # (N, G*L)
            mask = mask.view(b, G, L)
        else:
            # Independently mask each channel (i.e. spatial location has subset of channels visible)
            x, mask, ids_restore = self.random_masking(
                x.view(b, -1, D), mask_ratio
            )  # (N, 0.25*G*L, D)
            mask = mask.view(b, G, L)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, G*L + 1, D)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # (N, 1 + G*0.25*L, D)

        # append mask tokens to sequence
        G = len(self.channel_groups)
        if self.spatial_mask:
            N, L = ids_restore.shape

            x_ = (
                x[:, 1:, :].view(N, G, -1, x.shape[2]).permute(0, 2, 1, 3)
            )  # (N, 0.25*L, G, D)
            _, ml, _, D = x_.shape
            x_ = x_.reshape(N, ml, G * D)  # (N, 0.25*L, G*D)

            mask_tokens = self.mask_token.repeat(N, L - ml, G)
            x_ = torch.cat((x_, mask_tokens), dim=1)  # no cls token
            x_ = torch.gather(
                x_,
                dim=1,
                index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]),
            )  # (N, L, G*D)
            x_ = (
                x_.view(N, L, G, D).permute(0, 2, 1, 3).reshape(N, -1, D)
            )  # (N, G*L, D)
            x = torch.cat(
                (x[:, :1, :], x_), dim=1
            )  # append cls token  (N, 1 + G*L, D)
        else:
            mask_tokens = self.mask_token.repeat(
                x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
            )
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(
                x_,
                dim=1,
                index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
            )  # unshuffle
            x = torch.cat(
                [x[:, :1, :], x_], dim=1
            )  # append cls token  (N, 1 + c*L, D)

        # add pos and channel embed
        channel_embed = self.decoder_channel_embed[:, :-1, :].unsqueeze(
            2
        )  # (1, G, 1, cD)
        pos_embed = self.decoder_pos_embed[:, 1:, :].unsqueeze(
            1
        )  # (1, 1, L, pD)

        channel_embed = channel_embed.expand(
            -1, -1, pos_embed.shape[2], -1
        )  # (1, G, L, cD)
        pos_embed = pos_embed.expand(
            -1, channel_embed.shape[1], -1, -1
        )  # (1, G, L, pD)
        pos_channel = torch.cat(
            (pos_embed, channel_embed), dim=-1
        )  # (1, G, L, D)
        pos_channel = pos_channel.view(
            1, -1, pos_channel.shape[-1]
        )  # (1, G*L, D)

        extra = torch.cat(
            (
                self.decoder_pos_embed[:, :1, :],
                self.decoder_channel_embed[:, -1:, :],
            ),
            dim=-1,
        )  # (1, 1, D)

        pos_channel = torch.cat((extra, pos_channel), dim=1)  # (1, 1+G*L, D)
        x = x + pos_channel  # (N, 1+G*L, D)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # Separate channel axis
        N, GL, D = x.shape
        x = x.view(N, G, GL // G, D)

        # predictor projection
        x_c_patch = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, i]  # (N, L, D)
            dec = self.decoder_pred[i](x_c)  # (N, L, g_c * p^2)
            dec = dec.view(
                N, x_c.shape[1], -1, int(self.patch_size**2)
            )  # (N, L, g_c, p^2)
            dec = torch.einsum("nlcp->nclp", dec)  # (N, g_c, L, p^2)
            x_c_patch.append(dec)

        x = torch.cat(x_c_patch, dim=1)  # (N, c, L, p**2)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, c, H, W]
        pred: [N, L, c*p*p]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = patchify(
            imgs, self.patch_embed[0].patch_size[0], self.in_c
        )  # (N, L, C*P*P)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        N, L, _ = target.shape
        target = target.view(N, L, self.in_c, -1)  # (N, L, C, p^2)
        target = torch.einsum("nlcp->nclp", target)  # (N, C, L, p^2)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, C, L], mean loss per patch

        total_loss, num_removed = 0.0, 0.0
        current = 0
        for i, group in enumerate(self.channel_groups):
            interval = torch.arange(current, current + len(group))
            current += len(group)
            group_loss = loss[:, interval, :].mean(dim=1)  # (N, L)
            total_loss += (group_loss * mask[:, i]).sum()
            num_removed += mask[:, i].sum()  # mean loss on removed patches

        return total_loss / num_removed

    def forward(self, imgs, mask_ratio):
        # Group the images here
        x_c = []
        for i, group in enumerate(self.channel_groups):
            for band in group:
                x_c.append(imgs[:, band].unsqueeze(1))
        x_c = torch.cat(x_c, dim=1)
        imgs = x_c

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, C, L, p*p]
        return pred, mask


class SatVitNet(nn.Module):
    def __init__(
        self,
        num_channels=None,
        num_classes=4,
        segmentation_head=SimpleSegmentationHead,
        channel_embed=256,
        img_size=224,
        patch_size=8,
        embed_dim=1024,
        channel_groups=None,
        num_heads=16,
        depth=24,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        pretrained=True,
        pretrained_path=None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        if num_channels is not None:
            log.info("Num channels is ignore in profite of channel_groups")
        self.channel_groups = channel_groups
        self.num_channels = sum(len(group) for group in channel_groups)
        self.img_size = img_size

        self.model = SatViTEncoder(
            channel_embed=channel_embed,
            img_size=img_size,
            patch_size=patch_size,
            in_c=num_channels,
            embed_dim=embed_dim,
            channel_groups=channel_groups,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        if pretrained:
            state_dict = load_state_dict(pretrained_path, model_name="model")
            self.model.load_state_dict(state_dict, strict=False)

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


class SatMAENet(nn.Module):
    """ViTRoPE_MAE is a modified version of the ViTRoPE model that includes a Masked Autoencoder
    (MAE) decoder.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 4.
        num_channels (int, optional): Number of input channels. Defaults to 1.
        segmentation_head (nn.Module, optional): Segmentation head module(not initialized). Defaults to SimpleSegmentationHead.
        pretrained (bool, optional): Whether to use a pretrained model. Defaults to True.
        pretrained_path (str, optional): Path to the pretrained model. Defaults to None.
        img_size (int, optional): Size of the input image. Defaults to 512.
        ViT_patch_size (int, optional): Size of the ViT patches. Defaults to 16.
        ViT_embed_dim (int, optional): Dimension of the ViT embeddings. Defaults to 384.
        ViT_depth (int, optional): Depth of the ViT model. Defaults to 12.
        ViT_num_heads (int, optional): Number of heads in the ViT model. Defaults to 6.
        ViT_mlp_ratio (int, optional): Ratio of the MLP dimension to the ViT embedding dimension. Defaults to 4.
        MAE_depth (int, optional): Depth of the MAE decoder. Defaults to 3.
        MAE_drop_perc (float, optional): Percentage of patches to drop in the MAE decoder. Defaults to 0.75.
        rotary_position_emb (bool, optional): Whether to use rotary position embeddings. Defaults to True.
        conditionnal_posemb (bool, optional): Whether to use conditional position embeddings. Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        num_classes=4,
        num_channels=None,
        segmentation_head=SimpleSegmentationHead,
        channel_embed=256,
        img_size=512,
        patch_size=8,
        embed_dim=1024,
        channel_groups=None,
        num_heads=16,
        depth=24,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        decoder_channel_embed=128,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        MAE_drop_perc=0.75,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if num_channels is not None:
            log.info("Num channels is ignore in profite of channel_groups")
        self.channel_groups = channel_groups
        self.num_channels = sum(len(group) for group in channel_groups)
        self.img_size = img_size
        self.MAE_drop_prob = MAE_drop_perc
        self.patch_size = patch_size

        self.model = SatMAE(
            channel_embed=channel_embed,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_channels,
            embed_dim=embed_dim,
            channel_groups=channel_groups,
            num_heads=num_heads,
            depth=depth,
            decoder_channel_embed=decoder_channel_embed,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

    def forward(self, x, context=None, metas=None):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the output tensor.
        """
        pred, mask = self.model.forward(x, mask_ratio=self.MAE_drop_prob)
        # pred (N, c, L, p**2) , mask (N,L) 0 is keep, 1 is remove

        pred = pred.permute(0, 2, 1, 3)  # (N, L, c, p**2)
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1)

        pred = unpatchify(pred, self.model.patch_size, self.model.in_c)

        return {
            "out": pred,
            "mask": torch.logical_not(mask.squeeze().unsqueeze(-1)),
        }
