from shared import (
    ID, NAME, NB_EPOCHS, DATALOADER, BATCH_SIZE,
    TRAIN, VALIDATION, TEST,
    ARCHITECTURE, MODEL,
    N_PARAMS,
    OPTIMIZER, LR, PARAMS,
    SCHEDULER, REDUCELRONPLATEAU, SCHEDULER_CONFIGURATION
)
from model import UNet
import torch
from data_loader import get_dataloaders
from typing import Tuple
from shared import DEVICE


def get_experiment_config(exp: int) -> dict:
    config = {
        ID: exp,
        NAME: f"{exp:04d}",
        NB_EPOCHS: 200
    }
    config[DATALOADER] = {
        BATCH_SIZE: {
            TRAIN: 32,
            VALIDATION: 32,
            TEST: 32
        }
    }
    config[OPTIMIZER] = {
        NAME: "Adam",
        PARAMS: {
            LR: 1e-3
        }
    }
    config[MODEL] = {
        ARCHITECTURE: dict(
            ch_in=1,
            ch_out=1,
            num_layers=2,
            k_conv_ds=3,
            k_conv_us=3
        ),
        NAME: "UNet"
    }
    if exp == 0:
        config[NB_EPOCHS] = 5
    if exp == 1:
        config[NB_EPOCHS] = 200
        config[DATALOADER][BATCH_SIZE][TRAIN] = 92
    if exp == 2:
        config[NB_EPOCHS] = 200
        config[DATALOADER][BATCH_SIZE][TRAIN] = 92
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 92
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.5,
            "patience": 5
        }
    return config


def get_training_content(config: dict, device=DEVICE) -> Tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
    model = UNet(**config[MODEL][ARCHITECTURE])
    assert config[MODEL][NAME] == UNet.__name__
    if True:
        n, ch, h, w = 4, 1, 36, 36
        model(torch.rand(n, ch, w, h))
    config[MODEL][N_PARAMS] = model.count_parameters()
    optimizer = torch.optim.Adam(model.parameters(), **config[OPTIMIZER][PARAMS])
    dl_dict = get_dataloaders(config, device=device)
    return model, optimizer, dl_dict


if __name__ == "__main__":
    config = get_experiment_config(0)
    print(config)
    model, optimizer, dl_dict = get_training_content(config)
    print(len(dl_dict[TRAIN].dataset))