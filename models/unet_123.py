import torch

from models.sub_model import AblationUNet


class unet_123(AblationUNet):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            use_ar=True,
            use_wt=True,
            use_db=True,
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    model = unet_123(in_channels=3, num_classes=3, base_channels=32)
    x = torch.randn(1, 3, 64, 64)
    logits = model(x)
    print("logits:", logits['out'].shape)
