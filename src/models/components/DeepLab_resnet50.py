import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, deeplabv3_resnet50

from src.models.components.utils.utils import set_first_layer


class DeepLabResnet50(torch.nn.Module):
    def __init__(
        self,
        img_size=256,
        num_channels=1,
        num_classes=4,
        pretrained=False,
        pretrained_path=None,
        chkpt_path=None,
    ):
        super().__init__()

        if pretrained:
            self.model = deeplabv3_resnet50(
                weights=None, weights_backbone=None
            )
            weight = torch.load(pretrained_path)
            self.model.load_state_dict(weight, strict=False)
        else:
            self.model = deeplabv3_resnet50(
                weights=None, weights_backbone=None
            )

        if num_channels != 3:
            set_first_layer(self.model.backbone, num_channels, is_rgb=True)
        self.model.classifier = DeepLabHead(2048, num_classes)

        if chkpt_path:
            raise NotImplementedError("Checkpoint loading not implemented yet")

    def forward(self, x: torch.Tensor, metas=None) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # batch_size, channels, width, height = x.size()

        return self.model(x)


if __name__ == "__main__":
    _ = DeepLabResnet50()
