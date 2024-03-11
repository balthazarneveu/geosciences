import torch
from base_blocks import BaseConvolutionBlock
from model_base import BaseModel
from typing import Tuple, Optional, List


class ConvolutionStage(torch.nn.Module):
    def __init__(
        self,  ch_in: int, ch_out: int,
        h_dim: int = None,
        k_size: int = 3,
        bias: bool = True,
        activation="LeakyReLU",
        last_activation="LeakyReLU",
        depth: int = 1,
    ) -> None:
        """Chain several convolutions together to create a processing stage for 1 scale of the UNET
        """
        super().__init__()
        self.conv_stage = torch.nn.ModuleList()
        if h_dim is None:
            h_dim = max(ch_out, ch_in)
        for idx in range(depth):
            ch_inp = ch_in if idx == 0 else h_dim
            ch_outp = ch_out if idx == depth-1 else h_dim
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

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        depth: int = 1,
        k_size: int = 3,
        bias: bool = True,
        activation="LeakyReLU"
    ) -> None:

        super().__init__()

        self.conv_block = ConvolutionStage(ch_in, ch_out, k_size=k_size, bias=bias, activation=activation, depth=depth)

    def forward(self, x):
        x = self.conv_block(x)
        x_ds = x[..., ::2, ::2]
        return x, x_ds


class DecoderStage(torch.nn.Module):
    """Upsample by 2, Concatenate with skip connection, Conv (and shrink channels)
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        depth: int = 1,
        k_size: int = 3,
        bias: bool = True,
        activation="LeakyReLU"
    ) -> None:
        """Decoder stage
        """

        super().__init__()
        self.conv_block = ConvolutionStage(
            ch_in,
            ch_out,
            depth=depth,
            k_size=k_size,
            bias=bias,
            activation=activation,
        )
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
        refinement_stage_depth: int = 1,
        bottleneck_depth: int = 1,
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
        self.encoder_list.append(EncoderStage(ch_in, thickness, k_size=k_conv_ds, bias=bias, depth=encoders[0]))
        for level in range(1, num_layers+1):
            ch_i = (2**(level-1))*thickness
            ch_o = (2**(level))*thickness
            # print(ch_i, ch_o)
            if level < num_layers:
                # Skipping last encoder since we defined the first one outside the loop
                self.encoder_list.append(EncoderStage(ch_i, ch_o, k_size=k_conv_ds, bias=bias, depth=encoders[level]))
            self.decoder_list.append(DecoderStage(ch_o+ch_i, ch_i, k_size=k_conv_us,
                                     bias=bias, depth=decoders[num_layers-level]))
            if level == num_layers:
                self.bottleneck = ConvolutionStage(
                    ch_i, ch_o, depth=bottleneck_depth,
                    k_size=k_conv_ds, bias=bias
                )
        self.refinement_stage = ConvolutionStage(
            thickness,
            ch_out,
            k_size=1,
            depth=refinement_stage_depth,
            bias=bias,
            last_activation="Identity"
        )

    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward UNET pass
        Need to pad to 40x40 to get 4 scales
         ```
        (1  ,   40x40)                     REFINE      > (1  , 40x40)
        (16 ,   40x40)----------------->(16 ,   40x40)
            v                               ^
        (16 ,   20x20)----------------->(16 ,   20x20)
            v                               ^
        (32 ,   10x10)----------------->(32 ,   10x10)
            v                               ^
        (64 ,   5x5  )----BOTTLENECK--->(64 ,    5x5 )
        ```

        Since pytorch does not support circular padding per direction,
        We manually pad circularly on the horizontal direction and replicate on the vertical direction

        w x h                                w x h     [w,h]
        36x36 -> 108x36 -> (108+4)x(36+4) = 112x40 >>> 40,112

        extra=2 allows going to 40 which unlocks the use of 3 scales

        ---------
        If not stick to 36x36 and 3 scales

        ```
        (1  ,   36x36)                     REFINE      > (1  , 36x36)
        (16 ,   36x36)----------------->(16 ,   36x36)
            v                               ^
        (16 ,   18x18)----------------->(16 ,   18x18)
            v                               ^
        (32 ,   9x9  )----BOTTLENECK--->(32 ,    9x9 )
        ```
        """
        skipped_list = []
        x_padded = self.pad_input_tensor(x_in)
        ds_list = [x_padded]
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
        x_dec = self.crop_output_tensor(x_dec)
        out = self.refinement_stage(x_dec)
        return out

    @staticmethod
    def pad_input_tensor(x_in: torch.Tensor, w=36, extra=2) -> torch.Tensor:
        if x_in.shape[-1] != w:
            return x_in
        # print("PADDING", x_in.shape)
        x_h_pad = torch.nn.functional.pad(x_in, (w, w, w, w), "circular")
        x_h_pad = x_h_pad[..., w:-w, :]
        # print("PADDING", x_h_pad.shape)
        x_v_pad = torch.nn.functional.pad(x_h_pad, (extra, extra, extra, extra), "replicate")
        # print(f"PADDING", x_v_pad.shape)
        return x_v_pad

    @staticmethod
    def crop_output_tensor(x_out: torch.Tensor, w=36, extra=2) -> torch.Tensor:
        if x_out.shape[-1] == w:
            return x_out
        x_start = extra+w
        cropped = x_out[..., extra:-extra, x_start:x_start+w]
        # print("CROPPING", x_out.shape, cropped.shape)
        return cropped


def __check_cropping(plot_flag=False):
    x = torch.zeros(4, 1, 36, 36)
    x[:, :, 3:, 5:5+4] = 0.5
    x[:, :, -1, 5:5+4] = 1
    x[:, :, :10, -1:] = 0.2
    x[:, :, :10, 0] = 0.6

    y = FlexibleUNET.pad_input_tensor(x)
    z = FlexibleUNET.crop_output_tensor(y)

    assert torch.allclose(x, z), "Cropping and padding are not inverse operations"

    if plot_flag:
        print(x.shape, y.shape, z.shape)
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(x[0, 0].detach().numpy())
        plt.title("Original")
        plt.subplot(132)
        plt.imshow(z[0, 0].detach().numpy())
        plt.title("Cropped")
        plt.subplot(133)
        plt.imshow(y[0, 0].detach().numpy())
        plt.title(f"Padded\nx=circular y=replicate\n{y.shape}")
        plt.show()


def __check_flex_unet():
    model = FlexibleUNET(bias=True, encoders=[3, 1, 1], decoders=[1, 1, 3],
                         refinement_stage_depth=4, bottleneck_depth=1, thickness=32)
    print(model)
    print(f"Model #parameters {model.count_parameters()}")
    n, ch, h, w = 4, 1, 36, 36
    print(model(torch.rand(n, ch, w, h)).shape)
    receptive_x, receptive_y = model.receptive_field()
    print(f"UNET: Receptive field: x={receptive_x}  y={receptive_y}")


if __name__ == "__main__":
    __check_cropping(plot_flag=True)
    # __check_flex_unet()
