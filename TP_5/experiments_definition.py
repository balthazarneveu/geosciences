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
    elif exp == 30:  # Stacked convolutions
        experiment_stacked_convolutions(config, num_layers=5, h_dim=128, n=100)
    elif exp == 40:  # Classic UNet
        config[LOSS] = LOSS_DICE_BCE
        config[DATALOADER][BATCH_SIZE][TRAIN] = 128
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 128
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[MODEL][ARCHITECTURE]["channels_extension"] = 64
    elif exp == 50:  # Flexible UNet
        experiment_flexible_unet(
            config,
            n=100,
            b=128,
            lr=5e-4,
            loss=LOSS_DICE_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=16,
        )
    else:
        raise ValueError(f"Experiment {exp} not found")
    return config
