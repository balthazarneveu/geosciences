from shared import (
    ID, NAME, NB_EPOCHS, DATALOADER, BATCH_SIZE,
    TRAIN, VALIDATION, TEST,
    ARCHITECTURE, MODEL,
    N_PARAMS,
    OPTIMIZER, LR, PARAMS,
    SCHEDULER, REDUCELRONPLATEAU, SCHEDULER_CONFIGURATION,
    AUGMENTATION_LIST, AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP,
    LOSS, LOSS_BCE, LOSS_DICE, LOSS_BCE_WEIGHTED
)
from model import UNet, StackedConvolutions, VanillaConvolutionStack, MicroConv
import torch
from data_loader import get_dataloaders
from typing import Tuple
from shared import DEVICE


def experiment_stacked_convolutions(
    config: dict,
    b: int = 32,
    n: int = 50,
    h_dim: int = 256,
    num_layers: int = 5,
    k_conv_h: int = 3,
    k_conv_v: int = 5,
    residual: bool = False
) -> dict:
    config[NB_EPOCHS] = n
    config[DATALOADER][BATCH_SIZE][TRAIN] = b
    config[DATALOADER][BATCH_SIZE][VALIDATION] = b
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
            h_dim=h_dim,
            num_layers=num_layers,
            k_conv_h=k_conv_h,
            k_conv_v=k_conv_v,
            residual=residual
        ),
        NAME: "StackedConvolutions"
    }


def experiment_micro_conv(
    config: dict,
    b: int = 32,
    n: int = 50,
    h_dim: int = 4,
) -> dict:
    config[NB_EPOCHS] = n
    config[DATALOADER][BATCH_SIZE][TRAIN] = b
    config[DATALOADER][BATCH_SIZE][VALIDATION] = b
    config[SCHEDULER] = REDUCELRONPLATEAU
    config[SCHEDULER_CONFIGURATION] = {
        "factor": 0.8,
        "patience": 5
    }
    config[OPTIMIZER][PARAMS][LR] = 1e-3
    config[MODEL] = {
        ARCHITECTURE: dict(
            h_dim=h_dim,
        ),
        NAME: "MicroConv"
    }


def vanilla_experiment(config: dict, b: int = 32, n: int = 50) -> dict:
    config[NB_EPOCHS] = n
    config[DATALOADER][BATCH_SIZE][TRAIN] = b
    config[DATALOADER][BATCH_SIZE][VALIDATION] = b
    config[SCHEDULER] = REDUCELRONPLATEAU
    config[SCHEDULER_CONFIGURATION] = {
        "factor": 0.8,
        "patience": 5
    }
    config[OPTIMIZER][PARAMS][LR] = 1e-3
    config[MODEL] = {
        ARCHITECTURE: dict(),
        NAME: "VanillaConvolutionStack"
    }


def default_experiment(exp: int) -> dict:
    config = {
        ID: exp,
        NAME: f"{exp:04d}",
        NB_EPOCHS: 50
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
    config[LOSS] = LOSS_BCE
    return config


def get_experiment_config(exp: int) -> dict:
    config = default_experiment(exp)
    if exp == 0:
        config[NB_EPOCHS] = 5
    elif exp == 1:
        config[DATALOADER][BATCH_SIZE][TRAIN] = 256
    elif exp == 2:
        config[DATALOADER][BATCH_SIZE][TRAIN] = 92
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 92
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.5,
            "patience": 5
        }
    elif exp == 3:
        config[DATALOADER][BATCH_SIZE][TRAIN] = 32
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 32
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.5,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 8
    elif exp == 4:
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
        config[DATALOADER][BATCH_SIZE][TRAIN] = 32
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 32
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 32
        config[OPTIMIZER][PARAMS][LR] = 1e-4
    elif exp == 100:
        experiment_stacked_convolutions(config, num_layers=3, h_dim=64, n=50)
    elif exp == 101:
        experiment_stacked_convolutions(config, num_layers=5, h_dim=64, n=50)
    elif exp == 102:
        experiment_stacked_convolutions(config, num_layers=5, h_dim=32, n=50)
    elif exp == 103:
        experiment_stacked_convolutions(config, num_layers=3, h_dim=256, n=50)
    elif exp == 104:
        experiment_stacked_convolutions(config, num_layers=4, h_dim=256, n=50)
    elif exp == 105:
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
    elif exp == 106:
        experiment_stacked_convolutions(config, num_layers=6, h_dim=16, n=50, residual=True)
    elif exp == 107:
        experiment_stacked_convolutions(config, num_layers=6, h_dim=32, n=50, residual=True)
    elif exp == 108:
        experiment_stacked_convolutions(config, num_layers=6, h_dim=64, n=50, residual=True)
    elif exp == 109:
        experiment_stacked_convolutions(config, num_layers=8, h_dim=16, n=50, residual=True)
    elif exp == 110:
        # same as 105 but with augmentations
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED]
    elif exp == 111:
        # same as 105 but with augmentations
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_FLIP]
    elif exp == 112:
        # same as 105 but with augmentations
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    elif exp == 200:
        # same as 105 but with augmentations - new metrics
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE
    elif exp == 201:
        # same as 105 but with augmentations - new metrics  - first run coeff 9
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
    elif exp == 202:
        # same as 105 but with augmentations - new metrics
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_DICE
    elif exp == 203:
        # Same as 202 but with larger LR
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[OPTIMIZER][PARAMS][LR] = 1e-3
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_DICE
    elif exp == 204:
        # FIXED DICE same as 201
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
    elif exp == 300:
        vanilla_experiment(config, n=50, b=128)
        config[LOSS] = LOSS_BCE
    elif exp == 301:
        vanilla_experiment(config, n=50, b=128)
        config[LOSS] = LOSS_DICE
    elif exp == 302:
        vanilla_experiment(config, n=50, b=128)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_DICE
    elif exp == 303:
        vanilla_experiment(config, n=50, b=32)
        config[LOSS] = LOSS_BCE
    elif exp == 304:
        vanilla_experiment(config, n=50, b=32)
        config[LOSS] = LOSS_DICE
    elif exp == 305:
        vanilla_experiment(config, n=50, b=32)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_DICE
    elif exp == 306:  # similar as the stacked convolutions series
        vanilla_experiment(config, n=50, b=32)
        config[OPTIMIZER][PARAMS][LR] = 1e-4
        config[LOSS] = LOSS_BCE
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    elif exp == 306:
        # FIX DICE : SAME AS 300!
        vanilla_experiment(config, n=50, b=128)
        config[LOSS] = LOSS_BCE
    elif exp == 400:
        config[DATALOADER][BATCH_SIZE][TRAIN] = 32
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 32
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 32
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[OPTIMIZER][PARAMS][LR] = 1e-4
    elif exp == 401:
        config[DATALOADER][BATCH_SIZE][TRAIN] = 128
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 128
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 32
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[OPTIMIZER][PARAMS][LR] = 1e-4
    elif exp == 402:
        config[DATALOADER][BATCH_SIZE][TRAIN] = 128
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 128
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 64
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[OPTIMIZER][PARAMS][LR] = 1e-4
    elif exp == 403:
        # FIXED DICE : SAME AS 402!
        config[DATALOADER][BATCH_SIZE][TRAIN] = 128
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 128
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 64
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[OPTIMIZER][PARAMS][LR] = 1e-4
    elif exp == 500:
        experiment_micro_conv(config, h_dim=4, b=32, n=200)
    elif exp == 501:
        experiment_micro_conv(config, h_dim=4, b=32, n=200)
        config[OPTIMIZER][PARAMS][LR] = 1e-4
    elif exp == 502:
        experiment_micro_conv(config, h_dim=4, b=128, n=200)
        config[OPTIMIZER][PARAMS][LR] = 1e-3
    elif exp == 503:
        experiment_micro_conv(config, h_dim=4, b=512, n=200)
        config[OPTIMIZER][PARAMS][LR] = 1e-3
    else:
        raise ValueError(f"Unknown experiment {exp}")
    return config
    

def get_training_content(config: dict, device=DEVICE) -> Tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
    if config[MODEL][NAME] == UNet.__name__:
        model = UNet(**config[MODEL][ARCHITECTURE])
    elif config[MODEL][NAME] == MicroConv.__name__:
        model = MicroConv(**config[MODEL][ARCHITECTURE])
    elif config[MODEL][NAME] == StackedConvolutions.__name__:
        model = StackedConvolutions(**config[MODEL][ARCHITECTURE])
    elif config[MODEL][NAME] == VanillaConvolutionStack.__name__:
        model = VanillaConvolutionStack(**config[MODEL][ARCHITECTURE])
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
