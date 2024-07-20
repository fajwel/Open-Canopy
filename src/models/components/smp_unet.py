import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.utils.utils import set_first_layer


class SMP_Unet(nn.Module):
    """Pytorch segmentation U-Net with ResNet34 (default) with added metadata information at
    encoder output."""

    def __init__(
        self,
        img_size,
        num_channels,
        num_classes,
        pretrained_encoder=True,
        pretrained_encoder_path=None,
        chkpt_path=None,
    ):
        super().__init__()
        if pretrained_encoder and pretrained_encoder_path is None:
            encoder_weights = "imagenet"
        else:
            # if pretrained is False or path is provided, do not use pretrained weights
            encoder_weights = None

        self.seg_model = smp.create_model(
            arch="unet",
            encoder_name="resnet34",
            classes=num_classes,
            in_channels=3,
            encoder_weights=encoder_weights,
        )
        self.seg_model.encoder.load_state_dict(
            torch.load(pretrained_encoder_path)
        )
        set_first_layer(self.seg_model.encoder, num_channels)

        if chkpt_path:
            strict = True
            chkpt = torch.load(chkpt_path)["state_dict"]
            # remove 'model.seg_model.' from keys
            chkpt = {
                k.replace("model.seg_model.", ""): v for k, v in chkpt.items()
            }
            if chkpt["segmentation_head.0.weight"].shape[0] != num_classes:
                # if num_classes is different, remove segmentation_head weights
                chkpt = {
                    k: v
                    for k, v in chkpt.items()
                    if "segmentation_head" not in k
                }
                strict = False
                print(
                    "number of classes is different from checkpoint,  segmentation_head weights will be reinitialized"
                )
            if chkpt["encoder.conv1.weight"].shape[1] != num_channels:
                # if num_channels is different, remove encoder weights
                chkpt = {
                    k: v
                    for k, v in chkpt.items()
                    if "encoder.conv1.weight" not in k
                }
                strict = False
                print(
                    "number of channels is different from checkpoint,  encoder weights will be reinitialized"
                )
            self.seg_model.load_state_dict(chkpt, strict=strict)

    def forward(self, x, metas=None):
        """Forward function.

        Args:
            x (torch.tensor): input image

        Returns:
            Dict: dictionary with the output of the model as 'out'
        """
        output = self.seg_model(x)

        return {"out": output}
