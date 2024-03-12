from shared import (
    NAME,
    TRAIN,
    ARCHITECTURE, MODEL,
    N_PARAMS,
    OPTIMIZER, PARAMS,
)
from model import UNet, StackedConvolutions, VanillaConvolutionStack, MicroConv, FlexibleUNET
import torch
from data_loader import get_dataloaders
from typing import Tuple
from shared import DEVICE
from experiments_definition import get_experiment_config_latest
from experiments_legacy import get_experiment_config_legacy


def get_experiment_config(exp: int, legacy: bool = False) -> dict:
    if legacy:
        return get_experiment_config_legacy(exp)
    return get_experiment_config_latest(exp)


def get_training_content(config: dict, device=DEVICE, get_data_loaders_flag: str = True, total_freeze=False) -> Tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
    allowed_architectures = [UNet, MicroConv, StackedConvolutions, VanillaConvolutionStack, FlexibleUNET]
    allowed_architectures = {a.__name__: a for a in allowed_architectures}
    selected_architecture = allowed_architectures.get(config[MODEL][NAME], None)
    if selected_architecture is None:
        raise ValueError(f"Unknown model {config[MODEL][NAME]}")
    model = selected_architecture(**config[MODEL][ARCHITECTURE])
    if False:  # Sanity check on model
        n, ch, h, w = 4, 1, 36, 36
        model(torch.rand(n, ch, w, h))
    config[MODEL][N_PARAMS] = model.count_parameters()
    config[MODEL]["receptive_field"] = model.receptive_field()
    if get_data_loaders_flag:
        optimizer = torch.optim.Adam(model.parameters(), **config[OPTIMIZER][PARAMS])
        dl_dict = get_dataloaders(config, device=device, total_freeze=total_freeze)
    else:
        optimizer = None
        dl_dict = None
    return model, optimizer, dl_dict


if __name__ == "__main__":
    config = get_experiment_config(0)
    print(config)
    model, optimizer, dl_dict = get_training_content(config)
    print(len(dl_dict[TRAIN].dataset))
