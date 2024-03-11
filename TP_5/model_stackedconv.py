from model_base import BaseModel
from base_blocks import SpatialSplitConvolutionBlock, get_non_linearity
import torch


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
    # print(model(torch.rand(n, ch, w, h)).shape)
    n, ch, h, w = 4, 1, 36, 36
    print(model(torch.rand(n, ch, w, h)).shape)
    receptive_x, receptive_y = model.receptive_field()
    print(f"Receptive field: x={receptive_x}  y={receptive_y}")


if __name__ == "__main__":
    __check_convnet()
