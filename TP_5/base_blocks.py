import torch


def get_non_linearity(activation: str):
    if activation == "LeakyReLU":
        non_linearity = torch.nn.LeakyReLU()
    elif activation == "ReLU":
        non_linearity = torch.nn.ReLU()
    elif activation is None or activation == "Identity" or activation == "None":
        non_linearity = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    return non_linearity


class BaseConvolutionBlock(torch.nn.Module):
    def __init__(self, ch_in, ch_out: int, k_size: int, activation="LeakyReLU", bias: bool = True, padding_mode="zeros") -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(ch_in, ch_out, k_size, padding=k_size//2, bias=bias, padding_mode=padding_mode)
        self.non_linearity = get_non_linearity(activation)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv(x_in)  # [N, ch_in, H, W] -> [N, ch_in+channels_extension, H, W]
        x = self.non_linearity(x)
        return x


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
