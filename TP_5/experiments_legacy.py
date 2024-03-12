from shared import (
    NB_EPOCHS, DATALOADER, BATCH_SIZE,
    TRAIN, VALIDATION,
    ARCHITECTURE, MODEL,
    OPTIMIZER, LR, PARAMS,
    SCHEDULER, REDUCELRONPLATEAU, SCHEDULER_CONFIGURATION,
    AUGMENTATION_LIST, AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP,
    SYNTHETIC,
    LOSS, LOSS_BCE, LOSS_DICE, LOSS_BCE_WEIGHTED, LOSS_DICE_BCE,
    TRIVIAL, EASY, MEDIUM,
    DISTILLATION, DISTILLATION_CONFIG, TEACHER,
    TEMPERATURE, DISTILLATION_WEIGHT
)
from experiments_default import (
    experiment_stacked_convolutions, default_experiment, vanilla_experiment,
    experiment_micro_conv, experiment_flexible_unet
)


def get_experiment_config_legacy(exp: int) -> dict:
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
    elif exp == 206:
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50, b=128)
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
        config[NB_EPOCHS] = 200
    elif exp == 404:
        # FIXED DICE
        config[DATALOADER][BATCH_SIZE][TRAIN] = 128
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 128
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 64
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[OPTIMIZER][PARAMS][LR] = 1e-3
        config[NB_EPOCHS] = 200

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
    # TOY EXPERIMENTS
    elif exp == 600:  # EASY + STACKED
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = EASY
    elif exp == 601:  # EASY + UNET
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = EASY
    elif exp == 602:  # EASY + UNET
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = EASY
        config[NB_EPOCHS] = 300
    elif exp == 610:  # MODULATE + STACKED
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = MEDIUM
    elif exp == 611:  # MODULATE + UNET
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = MEDIUM
    elif exp == 620:  # TRIVIAL + STACKED
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = TRIVIAL
        config[NB_EPOCHS] = 70
        config[OPTIMIZER][PARAMS][LR] = 1e-3
    elif exp == 621 or exp == 625:  # TRIVIAL + UNET !!!!!!!!!!!!!!!!! 625 redo with fixed metrics
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = TRIVIAL
        config[NB_EPOCHS] = 100
        config[OPTIMIZER][PARAMS][LR] = 1e-3
    elif exp == 622:  # TRIVIAL + UNET - BIG BATCH
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = TRIVIAL
        config[NB_EPOCHS] = 100
        config[OPTIMIZER][PARAMS][LR] = 1e-3
        config[DATALOADER][BATCH_SIZE][TRAIN] = 256
    elif exp == 623:  # TRIVIAL + UNET + DICE LOSS
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_DICE
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = TRIVIAL
        config[NB_EPOCHS] = 100
        config[OPTIMIZER][PARAMS][LR] = 1e-3
    elif exp == 624:  # TRIVIAL + STACKED + DICE LOSS
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_DICE
        config[DATALOADER][SYNTHETIC] = True
        config[DATALOADER]["mode"] = TRIVIAL
        config[NB_EPOCHS] = 50
        config[OPTIMIZER][PARAMS][LR] = 1e-3
    # RESTART
    elif exp == 700:  # STACKED CONV
        # ~ 201/204
        experiment_stacked_convolutions(config, num_layers=5, h_dim=256, n=50)
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[LOSS] = LOSS_BCE_WEIGHTED
        config[NB_EPOCHS] = 100
    elif exp == 701:  # Micro CONV
        # ~ 503
        experiment_micro_conv(config, h_dim=4, b=32, n=200)
        config[OPTIMIZER][PARAMS][LR] = 1e-3
        config[NB_EPOCHS] = 100
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    elif exp == 702:  # UNET
        # ~ 402
        config[DATALOADER][BATCH_SIZE][TRAIN] = 64
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 64
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[MODEL][ARCHITECTURE]["channels_extension"] = 64
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[OPTIMIZER][PARAMS][LR] = 1e-4
        config[NB_EPOCHS] = 100
    elif exp == 703:  # UNET - extend 24 by default
        # ~ 402
        config[DATALOADER][BATCH_SIZE][TRAIN] = 64
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 64
        config[SCHEDULER] = REDUCELRONPLATEAU
        config[SCHEDULER_CONFIGURATION] = {
            "factor": 0.8,
            "patience": 5
        }
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
        config[OPTIMIZER][PARAMS][LR] = 1e-4
        config[NB_EPOCHS] = 100
    elif exp == 704:  # Flexible UNET
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=1e-3, loss=LOSS_BCE_WEIGHTED,
            encoders=[1, 1], decoders=[1, 1], thickness=16,
        )
    elif exp == 705:  # Flexible UNET
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=5e-4, loss=LOSS_DICE_BCE,
            encoders=[1, 1], decoders=[1, 1], thickness=16,
        )
    elif exp == 706:  # Flexible UNET B=32
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=5e-4, loss=LOSS_DICE_BCE,
            encoders=[1, 1], decoders=[1, 1], thickness=16,
        )
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    elif exp == 707:  # Flexible UNET B=64!
        experiment_flexible_unet(
            config,
            n=100, b=64, lr=5e-4, loss=LOSS_DICE_BCE,
            encoders=[1, 1], decoders=[1, 1], thickness=16,
        )
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    elif exp == 708:  # Flexible UNET 3 scales
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=5e-4, loss=LOSS_DICE_BCE,
            encoders=[1, 1, 1], decoders=[1, 1, 1], thickness=16,
        )
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    elif exp == 709:  # UNET
        experiment_flexible_unet(
            config,
            n=100, b=64, lr=5e-4, loss=LOSS_DICE_BCE,
            encoders=[1, 1, 1], decoders=[1, 1, 1], thickness=16,
        )
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    elif exp == 710:  # UNET
        experiment_flexible_unet(
            config,
            n=100, b=64, lr=5e-4, loss=LOSS_DICE_BCE,
            encoders=[1, 1, 1], decoders=[1, 1, 1], thickness=64,
        )
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    elif exp == 711:  # Bigger UNET
        experiment_flexible_unet(
            config,
            n=100, b=64, lr=5e-4, loss=LOSS_DICE_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=16,
        )
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]
    elif exp == 712 or exp == 713:  # Bigger UNET - 713 = REDO with fixed metrics
        experiment_flexible_unet(
            config,
            n=100, b=64, lr=5e-4, loss=LOSS_DICE_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=16, refinement_stage_depth=3
        )
        config[DATALOADER][AUGMENTATION_LIST] = [AUGMENTATION_H_ROLL_WRAPPED, AUGMENTATION_FLIP]

    elif exp == 900:
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=1e-3, loss=LOSS_BCE,
            encoders=[1, 1], decoders=[1, 1], thickness=4,
            refinement_stage_depth=1
        )
    elif exp == 1000:  # Flexible UNET -> Distill stacked conv!   ----------- T=2 - 0.8
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=1e-3, loss=LOSS_BCE,
            encoders=[1, 1], decoders=[1, 1], thickness=4,
            refinement_stage_depth=1
        )
        config[DISTILLATION] = True
        config[DISTILLATION_CONFIG] = {
            TEMPERATURE: 2.,
            DISTILLATION_WEIGHT: 0.8
        }
        # T=10 , 0.5 ! Stagne
        # T=2 , 0.8 ! Good!
        config[TEACHER] = 700
    elif exp == 1001:  # Flexible UNET -> Distill stacked conv!    ----------- T=4 - 0.8
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=1e-3, loss=LOSS_BCE,
            encoders=[1, 1], decoders=[1, 1], thickness=4,
            refinement_stage_depth=1
        )
        config[DISTILLATION] = True
        config[DISTILLATION_CONFIG] = {
            TEMPERATURE: 4.,
            DISTILLATION_WEIGHT: 0.8
        }
        config[TEACHER] = 700
    elif exp == 1002:  # Flexible UNET -> Distill stacked conv!   ----------- T=8 - 0.8
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=1e-3, loss=LOSS_BCE,
            encoders=[1, 1], decoders=[1, 1], thickness=4,
            refinement_stage_depth=1
        )
        config[DISTILLATION] = True
        config[DISTILLATION_CONFIG] = {
            TEMPERATURE: 8.,
            DISTILLATION_WEIGHT: 0.8
        }
        config[TEACHER] = 700
    elif exp == 1003:  # Flexible UNET -> Distill stacked conv!   ----------- T=2 - 0.5
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=1e-3, loss=LOSS_BCE,
            encoders=[1, 1], decoders=[1, 1], thickness=4,
            refinement_stage_depth=1
        )
        config[DISTILLATION] = True
        config[DISTILLATION_CONFIG] = {
            TEMPERATURE: 2.,
            DISTILLATION_WEIGHT: 0.5
        }
        config[TEACHER] = 700
    elif exp == 1004:  # Flexible UNET -> Distill stacked conv!   ----------- T=2 - 0.8 - Bigger UNET
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=1e-3, loss=LOSS_BCE,
            encoders=[4, 2, 1], decoders=[1, 2, 4], thickness=8,
            refinement_stage_depth=1
        )
        config[DISTILLATION] = True
        config[DISTILLATION_CONFIG] = {
            TEMPERATURE: 2.,
            DISTILLATION_WEIGHT: 0.8
        }
        config[TEACHER] = 700
    elif exp == 1005:  # Flexible UNET -> Distill stacked conv!   ----------- T=2 - 0.8 - Wider
        experiment_flexible_unet(
            config,
            n=100, b=32, lr=1e-3, loss=LOSS_BCE,
            encoders=[1, 1], decoders=[1, 1], thickness=16,
            refinement_stage_depth=1
        )
        config[DISTILLATION] = True
        config[DISTILLATION_CONFIG] = {
            TEMPERATURE: 2.,
            DISTILLATION_WEIGHT: 0.8
        }
        # T=10 , 0.5 ! Stagne
        # T=2 , 0.8 ! Good!
        config[TEACHER] = 700
    else:
        raise ValueError(f"Unknown experiment {exp}")
    return config
