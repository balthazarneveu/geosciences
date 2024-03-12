from shared import (
    ID, NAME, NB_EPOCHS, DATALOADER, BATCH_SIZE,
    TRAIN, VALIDATION, TEST,
    ARCHITECTURE, MODEL,
    N_PARAMS,
    OPTIMIZER, LR, PARAMS,
    SCHEDULER, REDUCELRONPLATEAU, SCHEDULER_CONFIGURATION,
    AUGMENTATION_LIST, AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP,
    SYNTHETIC,
    LOSS, LOSS_BCE, LOSS_DICE, LOSS_BCE_WEIGHTED, LOSS_DICE_BCE,
    TRIVIAL, EASY, MEDIUM, HARD,
    DISTILLATION, DISTILLATION_CONFIG, TEACHER,
    TEMPERATURE, DISTILLATION_WEIGHT
)
from model import UNet, StackedConvolutions, VanillaConvolutionStack, MicroConv, FlexibleUNET
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


def experiment_flexible_unet(
    config: dict,
    b: int = 32,
    n: int = 50,
    lr: float = 1e-3,
    encoders=[3, 1, 1],
    decoders=[1, 1, 3],
    thickness: int = 8,
    refinement_stage_depth=1,
    bottleneck_depth=1,
    loss=LOSS_BCE_WEIGHTED
) -> dict:
    config[MODEL] = {
        ARCHITECTURE: dict(
            # k_conv_ds=3,
            # k_conv_us=3,
            encoders=encoders,
            decoders=decoders,
            bottleneck_depth=bottleneck_depth,
            refinement_stage_depth=refinement_stage_depth,
            thickness=thickness,
            padding=True,
            bias=True,
        ),
        NAME: "FlexibleUNET"
    }
    config[NB_EPOCHS] = n
    config[DATALOADER][BATCH_SIZE][TRAIN] = b
    config[DATALOADER][BATCH_SIZE][VALIDATION] = b
    config[OPTIMIZER][PARAMS][LR] = lr
    config[SCHEDULER] = REDUCELRONPLATEAU
    config[SCHEDULER_CONFIGURATION] = {
        "factor": 0.8,
        "patience": 5
    }
    config[LOSS] = loss


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
