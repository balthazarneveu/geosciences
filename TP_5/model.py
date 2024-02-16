import torch
from typing import Tuple, Optional


class BaseModel(torch.nn.Module):
    """Base class for all models with additional methods"""

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_non_linearity(activation: str):
    if activation == "LeakyReLU":
        non_linearity = torch.nn.LeakyReLU()
    else:
        non_linearity = torch.nn.ReLU()
    return non_linearity


class BaseConvolutionBlock(torch.nn.Module):
    def __init__(self, ch_in, ch_out: int, k_size: int, activation="LeakyReLU", bias: bool = True) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(ch_in, ch_out, k_size, padding=k_size//2, bias=bias)
        self.non_linearity = get_non_linearity(activation)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv(x_in)  # [N, ch_in, H, W] -> [N, ch_in+channels_extension, H, W]
        x = self.non_linearity(x)
        return x


class EncoderStage(torch.nn.Module):
    """Conv (and extend channels), downsample 2 by skipping samples
    """

    def __init__(self, ch_in: int, ch_out: int, k_size: int = 15, bias: bool = True) -> None:

        super().__init__()

        self.conv_block = BaseConvolutionBlock(ch_in, ch_out, k_size=k_size, bias=bias)

    def forward(self, x):
        x = self.conv_block(x)
        x_ds = x[..., ::2, ::2]
        return x, x_ds


class DecoderStage(torch.nn.Module):
    """Upsample by 2, Concatenate with skip connection, Conv (and shrink channels)
    """

    def __init__(self, ch_in: int, ch_out: int, k_size: int = 5, dropout: float = 0., bias: bool = True) -> None:
        """Decoder stage
        """

        super().__init__()
        self.conv_block = BaseConvolutionBlock(ch_in, ch_out, k_size=k_size, bias=bias)
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x_ds: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """"""
        x_us = self.upsample(x_ds)  # [N, C, H/2, W/2] -> [N, C, H, W]
        print(x_us.shape, x_skip.shape)
        x = torch.cat([x_us, x_skip], dim=1)  # [N, 2C, H, W]
        x = self.conv_block(x)  # [N, C, H, W]
        return x


class UNet(BaseModel):
    """UNET in spatial domain (image of a well)
    """

    def __init__(self,
                 ch_in: int = 1,
                 ch_out: int = 1,
                 channels_extension: int = 24,
                 k_conv_ds: int = 3,
                 k_conv_us: int = 3,
                 num_layers: int = 2,
                 bias: bool = True,
                 ) -> None:
        super().__init__()
        self.ch_out = ch_out
        self.encoder_list = torch.nn.ModuleList()
        self.decoder_list = torch.nn.ModuleList()
        # Defining first encoder
        self.encoder_list.append(EncoderStage(ch_in, channels_extension, k_size=k_conv_ds, bias=bias))
        for level in range(1, num_layers+1):
            ch_i = level*channels_extension
            ch_o = (level+1)*channels_extension
            if level < num_layers:
                # Skipping last encoder since we defined the first one outside the loop
                self.encoder_list.append(EncoderStage(ch_i, ch_o, k_size=k_conv_ds, bias=bias))
            self.decoder_list.append(DecoderStage(ch_o+ch_i, ch_i, k_size=k_conv_us, bias=bias))
        self.bottleneck = BaseConvolutionBlock(
            num_layers*channels_extension,
            (num_layers+1)*channels_extension,
            k_size=k_conv_ds,
            bias=bias)
        self.target_modality_conv = torch.nn.Conv2d(
            channels_extension+ch_in, ch_out, 1, bias=bias)  # conv1x1 channel mixer

    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward UNET pass

        ```
        (1  ,   28)----------------->(24 , 28) > (1  , 28)
            v                            ^
        (24 ,   14)----------------->(48 ,   14)
            v                            ^
        (48 ,   7 )----BOTTLENECK--->(72,    7 )
        ```
        """
        skipped_list = []
        ds_list = [x_in]
        for level, enc in enumerate(self.encoder_list):
            x_skip, x_ds = enc(ds_list[-1])
            skipped_list.append(x_skip)
            ds_list.append(x_ds.clone())
        x_dec = self.bottleneck(ds_list[-1])
        for level, dec in enumerate(self.decoder_list[::-1]):
            print("Decoder in", x_dec.shape, skipped_list[-1-level].shape)
            x_dec = dec(x_dec, skipped_list[-1-level])
        x_dec = torch.cat([x_dec, x_in], dim=1)
        out = self.target_modality_conv(x_dec)
        return out


if __name__ == "__main__":
    model = UNet(bias=True, num_layers=2)
    print(f"Model #parameters {model.count_parameters()}")
    n, ch, h, w = 4, 1, 28, 28
    print(model(torch.rand(n, ch, w, h)).shape)
