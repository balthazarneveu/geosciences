import torch
from model_base import BaseModel


class MicroConv(BaseModel):
    """Micro Model 1.6k parameters proposed by Heni
    """

    def __init__(
        self,
        ch_in: int = 1,
        ch_out: int = 1,
        h_dim: int = 4,
        k_size: int = 3,
        padding: str = "same"
    ) -> None:
        super().__init__()
        h2dim = 2*h_dim
        self.conv1 = torch.nn.Conv2d(in_channels=ch_in, out_channels=h_dim, kernel_size=k_size, padding=padding)
        self.conv2 = torch.nn.Conv2d(in_channels=h_dim, out_channels=h_dim, kernel_size=k_size, padding=padding)

        self.conv3 = torch.nn.Conv2d(in_channels=h_dim, out_channels=h2dim, kernel_size=k_size, padding=padding)
        self.conv4 = torch.nn.Conv2d(in_channels=h2dim, out_channels=h2dim, kernel_size=k_size, padding=padding)

        self.up1 = torch.nn.ConvTranspose2d(in_channels=h2dim, out_channels=h_dim, kernel_size=2, stride=2)

        self.conv5 = torch.nn.Conv2d(in_channels=h2dim, out_channels=h_dim, kernel_size=k_size, padding=padding)
        self.conv6 = torch.nn.Conv2d(in_channels=h_dim, out_channels=h_dim, kernel_size=k_size, padding=padding)
        self.conv7 = torch.nn.Conv2d(in_channels=h_dim, out_channels=ch_out,
                                     kernel_size=1, padding=padding)  # pointwise
        self.non_linearity = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.non_linearity(self.conv1(x))
        x1 = self.non_linearity(self.conv2(x1))
        x2 = self.maxpool(x1)
        x2 = self.non_linearity(self.conv3(x2))
        x2 = self.non_linearity(self.conv4(x2))
        x = self.up1(x2)
        x = torch.cat([x1, x], dim=1)
        x = self.non_linearity(self.conv5(x))
        x = self.non_linearity(self.conv6(x))
        x = self.conv7(x)
        return x


def __check_microseg():
    model = MicroConv()
    print(model)
    print(f"Model #parameters {model.count_parameters()}")
    n, ch, h, w = 4, 1, 36, 36
    receptive_x, receptive_y = model.receptive_field()
    print(f"Receptive field: x={receptive_x}  y={receptive_y}")
    print(model(torch.rand(n, ch, w, h)).shape)


if __name__ == "__main__":
    __check_microseg()
