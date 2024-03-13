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
from experiments_default import (
    experiment_stacked_convolutions, default_experiment, vanilla_experiment,
    experiment_micro_conv, experiment_flexible_unet
)


def get_experiment_config_latest(exp: int) -> dict:
    config = default_experiment(exp)
    config[SCHEDULER_CONFIGURATION] = {
        "factor": 0.8,
        "patience": 10
    }
    config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    config[OPTIMIZER][PARAMS][LR] = 1e-3
    config[NB_EPOCHS] = 100
    config[LOSS] = LOSS_DICE_BCE
    if exp == 10:  # Vanilla experiment
        vanilla_experiment(config, n=100, b=128)
    elif exp == 20:  # MicoConv experiment
        experiment_micro_conv(config, n=100, b=128, h_dim=4)
    elif exp == 30:  # Stacked convolutions - 477k
        experiment_stacked_convolutions(config, b=32, num_layers=5, h_dim=128, n=100)
    elif exp == 31:  # Stacked convolutions - 477k - bigger batches
        experiment_stacked_convolutions(config, b=128, num_layers=5, h_dim=128, n=100)
    elif exp == 32:  # Stacked convolutions - 1.9M
        experiment_stacked_convolutions(config, b=32, num_layers=5, h_dim=256, n=100)
    elif exp == 40:  # Classic UNet
        config[DATALOADER][BATCH_SIZE][TRAIN] = 128
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 128
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[MODEL][ARCHITECTURE]["channels_extension"] = 64
    elif exp == 50:  # Flexible [4211124]UNet T16 404k parameters -> 73.9% dice
        experiment_flexible_unet(
            config,
            n=100,
            b=128,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=16,
        )
    elif exp == 51:  # Flexible [4211124]UNet T4 -> 71% dice
        experiment_flexible_unet(
            config,
            n=100,
            b=128,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=4,
        )
    elif exp == 52:  # Flexible [42124]UNet
        experiment_flexible_unet(
            config,
            n=100,
            b=128,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[4, 2], decoders=[2, 4], thickness=16,
        )
    elif exp == 53:  # Flexible [4211124]UNet T64 -> 79% dice  6.4M parameters - close to the 16Gb memory limit ***
        experiment_flexible_unet(
            config,
            n=100,
            b=128,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=64,
        )
    elif exp == 54:  # Flexible [4211124]UNet T16 - batch32 404k parameters
        experiment_flexible_unet(
            config,
            n=100,
            b=32,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=16,
        )
    elif exp == 55:  # Flexible [8421248]UNet T16 1.04M parameters
        experiment_flexible_unet(
            config,
            n=100,
            b=128,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[8, 4, 2], decoders=[2, 4, 8], thickness=16,
        )
    # Exp 70 series: Improve exp 55
    elif exp == 70:  # Flexible [8421248]UNet T16 1.04M parameters - 300 epochs
        experiment_flexible_unet(
            config,
            n=300,
            b=128,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[8, 4, 2], decoders=[2, 4, 8], thickness=16,
            refinement_stage_depth=4,
        )
    elif exp == 71:  # Flexible [8421248]UNet T16 1.04M parameters - 300 epochs
        experiment_flexible_unet(
            config,
            n=300,
            b=128,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[16, 8, 4], decoders=[4, 8, 16], thickness=16
        )

    # Exp 60 series: Improve exp 53
    elif exp == 60:  # Flexible [4211124]UNet T64 dice  6.4M parameters - 100 epochs - lr e-3
        experiment_flexible_unet(
            config,
            n=300,
            b=128,
            lr=1e-3,
            loss=LOSS_DICE_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=64,
        )
    elif exp == 61:  # Flexible [4211124]UNet T64  6.4M parameters - 300 epochs - lr 5e-4
        experiment_flexible_unet(
            config,
            n=300,
            b=128,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=64,
        )
    elif exp == 62:  # Flexible [4211124]UNet T64  6.4M parameters - 300 epochs - lr 5e-4 -> BCE weighted
        experiment_flexible_unet(
            config,
            n=300,
            b=128,
            lr=5e-4,
            loss=LOSS_BCE_WEIGHTED,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=64,
        )
    else:
        raise ValueError(f"Experiment {exp} not found")
    return config
