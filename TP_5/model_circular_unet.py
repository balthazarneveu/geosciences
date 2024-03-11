import torch
from base_blocks import BaseConvolutionBlock
from model_base import BaseModel
from typing import Tuple, Optional, List


class ConvolutionStage(torch.nn.Module):
    def __init__(
        self,  ch_in: int, ch_out: int, k_size: int = 3, bias: bool = True,
        activation="LeakyReLU",
        last_activation="LeakyReLU",
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.conv_stage = torch.nn.ModuleList()
        for idx in range(depth):
            ch_inp = ch_in if idx == 0 else ch_out
            ch_outp = ch_out
            self.conv_stage.append(
                BaseConvolutionBlock(
                    ch_inp, ch_outp,
                    k_size=k_size,
                    activation=activation if idx < depth-1 else last_activation,
                    bias=bias
                )
            )
        self.conv_stage = torch.nn.Sequential(*self.conv_stage)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.conv_stage(x_in)


class EncoderStage(torch.nn.Module):
    """Conv (and extend channels), downsample 2 by skipping samples
    """

    def __init__(self, ch_in: int, ch_out: int, k_size: int = 3, bias: bool = True, activation="LeakyReLU") -> None:

        super().__init__()

        self.conv_block = BaseConvolutionBlock(ch_in, ch_out, k_size=k_size, bias=bias, activation=activation)

    def forward(self, x):
        x = self.conv_block(x)
        x_ds = x[..., ::2, ::2]
        return x, x_ds


class DecoderStage(torch.nn.Module):
    """Upsample by 2, Concatenate with skip connection, Conv (and shrink channels)
    """

    def __init__(self, ch_in: int, ch_out: int, k_size: int = 3, bias: bool = True, activation="LeakyReLU") -> None:
        """Decoder stage
        """

        super().__init__()
        self.conv_block = BaseConvolutionBlock(ch_in, ch_out, k_size=k_size, bias=bias)
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x_ds: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """"""
        x_us = self.upsample(x_ds)  # [N, C, H/2, W/2] -> [N, C, H, W]
        x = torch.cat([x_us, x_skip], dim=1)  # [N, 2C, H, W]
        x = self.conv_block(x)  # [N, C, H, W]
        return x


class FlexibleUNET(BaseModel):
    def __init__(
        self,
        ch_in: int = 1,
        ch_out: int = 1,
        k_conv_ds: int = 3,
        k_conv_us: int = 3,
        thickness: int = 16,
        encoders: List[int] = (1, 1, 1),
        decoders: List[int] = (1, 1, 1),
        bottleneck: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        super().__init__()
        self.ch_out = ch_out
        self.encoder_list = torch.nn.ModuleList()
        self.decoder_list = torch.nn.ModuleList()
        # Defining first encoder
        assert len(encoders) == len(decoders), "Number of encoders and decoders should be the same"
        num_layers = len(encoders)
        self.encoder_list.append(EncoderStage(ch_in, thickness, k_size=k_conv_ds, bias=bias))
        for level in range(1, num_layers+1):
            ch_i = (2**(level-1))*thickness
            ch_o = (2**(level))*thickness
            print(ch_i, ch_o)
            if level < num_layers:
                # Skipping last encoder since we defined the first one outside the loop
                self.encoder_list.append(EncoderStage(ch_i, ch_o, k_size=k_conv_ds, bias=bias))
            self.decoder_list.append(DecoderStage(ch_o+ch_i, ch_i, k_size=k_conv_us, bias=bias))
            if level == num_layers:
                self.bottleneck = BaseConvolutionBlock(
                    ch_i, ch_o, k_size=k_conv_ds, bias=bias)
        self.target_modality_conv = BaseConvolutionBlock(
            thickness, ch_out, k_size=1, bias=bias, activation="Identity"
        )

    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward UNET pass

        ```
        (1  ,   36x36)
        (16 ,   36x36)----------------->(16 ,   36x36) > (1  , 36x36)
            v                               ^
        (16 ,   18x18)----------------->(16 ,   18x18)
            v                               ^
        (32 ,   9x9  )----BOTTLENECK--->(32 ,    9x9 )
        ```
        """
        skipped_list = []
        ds_list = [x_in]
        for level, enc in enumerate(self.encoder_list):
            x_skip, x_ds = enc(ds_list[-1])
            skipped_list.append(x_skip)
            ds_list.append(x_ds.clone())
            # print("Encoding", level, x_skip.shape, x_ds.shape)
        x_dec = self.bottleneck(ds_list[-1])
        # print("Bottleneck", x_dec.shape)
        for level, dec in enumerate(self.decoder_list[::-1]):
            # in_shape = x_dec.shape
            x_dec = dec(x_dec, skipped_list[-1-level])
            # print("Decoding", level, "(", in_shape, "+", skipped_list[-1-level].shape, ") ->", x_dec.shape)

        out = self.target_modality_conv(x_dec)
        return out


def __check_flex_unet():
    model = FlexibleUNET(bias=True, encoders=[1, 1], decoders=[1, 1])
    print(model)
    print(f"Model #parameters {model.count_parameters()}")
    # n, ch, h, w = 4, 1, 28, 28
    # print(model(torch.rand(n, ch, w, h)).shape)
    n, ch, h, w = 4, 1, 36, 36
    print(model(torch.rand(n, ch, w, h)).shape)
    receptive_x, receptive_y = model.receptive_field()
    print(f"UNET: Receptive field: x={receptive_x}  y={receptive_y}")


if __name__ == "__main__":
    __check_flex_unet()
