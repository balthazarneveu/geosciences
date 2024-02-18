from shared import (
    ID, NAME, NB_EPOCHS, DATALOADER, BATCH_SIZE,
    TRAIN, VALIDATION, TEST,
    ARCHITECTURE, MODEL,
    N_PARAMS,
    OPTIMIZER, LR, PARAMS,
    SCHEDULER, REDUCELRONPLATEAU, SCHEDULER_CONFIGURATION
)
from model import UNet, BaseCNN
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
    elif exp == 1:
        config[NB_EPOCHS] = 50
        config[DATALOADER][BATCH_SIZE][TRAIN] = 256
    elif exp == 2:
        config[NB_EPOCHS] = 50
        config[DATALOADER][BATCH_SIZE][TRAIN] = 92
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 92
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.5,
            "patience": 5
        }
    elif exp == 3:
        config[NB_EPOCHS] = 50
        config[DATALOADER][BATCH_SIZE][TRAIN] = 32
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 32
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.5,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 8
    elif exp == 4:
        config[NB_EPOCHS] = 50
        config[DATALOADER][BATCH_SIZE][TRAIN] = 32
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 32
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.5,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 32
        config[OPTIMIZER][PARAMS][LR] = 1e-4
    elif exp == 5:
        config[NB_EPOCHS] = 50
        config[DATALOADER][BATCH_SIZE][TRAIN] = 92
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 92
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.5,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 32
        config[OPTIMIZER][PARAMS][LR] = 1e-4
    elif exp == 6:
        config[NB_EPOCHS] = 50
        config[DATALOADER][BATCH_SIZE][TRAIN] = 32
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 32
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 32
        config[OPTIMIZER][PARAMS][LR] = 1e-4
    elif exp == 7:
        config[NB_EPOCHS] = 50
        config[DATALOADER][BATCH_SIZE][TRAIN] = 32
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 32
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[OPTIMIZER][PARAMS][LR] = 1e-4
        config[MODEL] = {
            ARCHITECTURE: dict(
                ch_in=1,
                ch_out=1,
                h_dim=256,
                k_conv_h=3,
                k_conv_v=5
            ),
            NAME: "BaseCNN"
        }
    return config


def get_training_content(config: dict, device=DEVICE) -> Tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
    if config[MODEL][NAME] == UNet.__name__:
        model = UNet(**config[MODEL][ARCHITECTURE])
    elif config[MODEL][NAME] == BaseCNN.__name__:
        model = BaseCNN(**config[MODEL][ARCHITECTURE])
    else:
        raise ValueError(f"Unknown model {config[MODEL][NAME]}")
    if False:  # Sanity check on model
        n, ch, h, w = 4, 1, 36, 36
        model(torch.rand(n, ch, w, h))
    config[MODEL][N_PARAMS] = model.count_parameters()
    config[MODEL]["receptive_field"] = model.receptive_field()
    optimizer = torch.optim.Adam(model.parameters(), **config[OPTIMIZER][PARAMS])
    dl_dict = get_dataloaders(config, device=device)
    return model, optimizer, dl_dict


if __name__ == "__main__":
    config = get_experiment_config(0)
    print(config)
    model, optimizer, dl_dict = get_training_content(config)
    print(len(dl_dict[TRAIN].dataset))
