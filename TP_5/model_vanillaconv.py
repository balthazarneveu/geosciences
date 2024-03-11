from model_base import BaseModel
from base_blocks import BaseConvolutionBlock
import torch


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
    # print(model(torch.rand(n, ch, w, h)).shape)
    n, ch, h, w = 4, 1, 36, 36
    print(model(torch.rand(n, ch, w, h)).shape)
    receptive_x, receptive_y = model.receptive_field()
    print(f"Receptive field: x={receptive_x}  y={receptive_y}")


if __name__ == "__main__":
    __check_convnet_vanilla()
