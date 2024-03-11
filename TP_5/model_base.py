import torch


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
