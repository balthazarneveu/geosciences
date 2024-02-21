import torch
from typing import Tuple, Optional


class BaseModel(torch.nn.Module):
    """Base class for all models with additional methods"""

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(self) -> int:
        """Compute the receptive field of the model

        Returns:
            int: receptive field
        """
        input_tensor = torch.rand(1, 1, 128, 128, requires_grad=True)
        out = self.forward(input_tensor)
        grad = torch.zeros_like(out)
        grad[..., out.shape[-2]//2, out.shape[-1]//2] = torch.nan  # set NaN gradient at the middle of the output
        out.backward(gradient=grad)
        self.zero_grad()  # reset to avoid future problems
        receptive_field_mask = input_tensor.grad.isnan()[0, 0]
        # print(receptive_field_mask)
        receptive_field_indexes = torch.where(receptive_field_mask)
        # print(receptive_field_indexes)
        # Count NaN in the input
        receptive_x = 1+receptive_field_indexes[-1].max() - receptive_field_indexes[-1].min()  # Horizontal x
        receptive_y = 1+receptive_field_indexes[-2].max() - receptive_field_indexes[-2].min()  # Vertical y
        # print(receptive_field_indexes)
        return receptive_x, receptive_y


class SpatialSplitConvolutionBlock(torch.nn.Module):
    def __init__(
        self, ch_in: int, ch_out: int,
        k_size_h: int = 3,
        k_size_v: int = 3,
        activation=None,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.conv_h = torch.nn.Conv2d(ch_in, ch_out, (1, k_size_h),
                                      padding=(0, k_size_h // 2),
                                      bias=bias, padding_mode="circular")
        self.conv_v = torch.nn.Conv2d(ch_out, ch_out, (k_size_v, 1),
                                      padding=(k_size_v //
                                      2, 0), bias=bias, padding_mode="replicate")
        if activation is None:
            self.non_linearity = torch.nn.Identity()
        else:
            self.non_linearity = get_non_linearity(activation)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv_h(x_in)
        x = self.conv_v(x)
        x = self.non_linearity(x)
        return x


class VanillaConvolutionStack(BaseModel):
    def __init__(self,
                 ch_in: int = 1,
                 ch_out: int = 1,
                 h_dim: int = [64, 64, 128, 128],
                 activation: str = "ReLU"
                 ) -> None:
        super().__init__()
        conv_dimensions = [ch_in] + h_dim + [ch_out]
        conv_list = []
        num_layers = len(conv_dimensions)-1
        for idx in range(num_layers):
            ch_inp, ch_outp = conv_dimensions[idx], conv_dimensions[idx+1]
            conv_list.append(BaseConvolutionBlock(ch_inp, ch_outp, k_size=3,
                             activation=None if idx == num_layers-1 else activation, bias=True))
        self.conv_stack = torch.nn.Sequential(*conv_list)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x_in)
        return x


def __check_convnet_vanilla():
    model = VanillaConvolutionStack()
    print(model)
    print(f"Model #parameters {model.count_parameters()}")
    # n, ch, h, w = 4, 1, 28, 28
    # print(model(torch.rand(n, ch, w, h)).shape)
    n, ch, h, w = 4, 1, 36, 36
    print(model(torch.rand(n, ch, w, h)).shape)
    receptive_x, receptive_y = model.receptive_field()
    print(f"Receptive field: x={receptive_x}  y={receptive_y}")


class StackedConvolutions(BaseModel):
    def __init__(self,
                 ch_in: int = 1,
                 ch_out: int = 1,
                 h_dim: int = 64,
                 num_layers: int = 3,
                 k_conv_h: int = 3,
                 k_conv_v: int = 5,
                 activation: str = "LeakyReLU",
                 bias: bool = True,
                 residual: bool = False
                 ) -> None:
        super().__init__()
        self.residual = residual
        self.conv_in_modality = SpatialSplitConvolutionBlock(
            ch_in, h_dim, k_size_h=k_conv_h, k_size_v=k_conv_v, activation=activation, bias=bias)
        if not residual:
            conv_list = []
            for _i in range(num_layers-2):
                conv_list.append(SpatialSplitConvolutionBlock(
                    h_dim, h_dim, k_size_h=k_conv_h, k_size_v=k_conv_v, activation=activation, bias=bias))
            self.conv_stack = torch.nn.Sequential(*conv_list)
        else:
            num_temp_layers = num_layers-2
            assert num_temp_layers % 2 == 0, "Number of layers should be even"
            self.conv_stack = torch.nn.modules.ModuleList()
            for _i in range(num_temp_layers//2):
                for activ in [activation, None]:
                    self.conv_stack.append(
                        SpatialSplitConvolutionBlock(
                            h_dim, h_dim, k_size_h=k_conv_h, k_size_v=k_conv_v, activation=activ, bias=bias))
            self.non_linearity = get_non_linearity(activation)
        self.conv_out_modality = SpatialSplitConvolutionBlock(
            h_dim, ch_out, k_size_h=k_conv_h, k_size_v=k_conv_v, activation=None, bias=bias)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv_in_modality(x_in)
        if self.residual:
            for idx in range(len(self.conv_stack)//2):
                y = self.conv_stack[2*idx](x)
                y = self.conv_stack[2*idx+1](y)
                x = x + y
                x = self.non_linearity(x)
        else:
            x = self.conv_stack(x)
        x = self.conv_out_modality(x)
        return x


def __check_convnet():
    model = StackedConvolutions(bias=True, residual=True, num_layers=6, h_dim=16)
    print(model)
    print(f"Model #parameters {model.count_parameters()}")
    # n, ch, h, w = 4, 1, 28, 28
    # print(model(torch.rand(n, ch, w, h)).shape)
    n, ch, h, w = 4, 1, 36, 36
    print(model(torch.rand(n, ch, w, h)).shape)
    receptive_x, receptive_y = model.receptive_field()
    print(f"Receptive field: x={receptive_x}  y={receptive_y}")


def get_non_linearity(activation: str):
    if activation == "LeakyReLU":
        non_linearity = torch.nn.LeakyReLU()
    elif activation == "ReLU":
        non_linearity = torch.nn.ReLU()
    elif activation is None:
        non_linearity = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
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
            x_dec = dec(x_dec, skipped_list[-1-level])
        x_dec = torch.cat([x_dec, x_in], dim=1)
        out = self.target_modality_conv(x_dec)
        return out


def __check_unet():
    model = UNet(bias=True, num_layers=2)
    print(f"Model #parameters {model.count_parameters()}")
    # n, ch, h, w = 4, 1, 28, 28
    # print(model(torch.rand(n, ch, w, h)).shape)
    n, ch, h, w = 4, 1, 36, 36
    print(model(torch.rand(n, ch, w, h)).shape)
    receptive_x, receptive_y = model.receptive_field()
    print(f"UNET: Receptive field: x={receptive_x}  y={receptive_y}")


if __name__ == "__main__":
    __check_convnet_vanilla()
    __check_convnet()
    __check_unet()
