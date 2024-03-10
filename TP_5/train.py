import sys
import argparse
from typing import Optional
import torch
import logging
from pathlib import Path
import json
from tqdm import tqdm
from shared import (
    ROOT_DIR, OUTPUT_FOLDER_NAME,
    ID, NAME, NB_EPOCHS,
    TRAIN, VALIDATION, TEST, LR,
    ACCURACY, PRECISION, RECALL, F1_SCORE, IOU,
    DEVICE, SCHEDULER_CONFIGURATION, SCHEDULER, REDUCELRONPLATEAU,
    LOSS, LOSS_BCE
)
from metrics import compute_metrics
from loss import compute_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from configuration import WANDBSPACE
from experiments import get_experiment_config, get_training_content
WANDB_AVAILABLE = False
try:
    WANDB_AVAILABLE = True
    import wandb
except ImportError:
    logging.warning("Could not import wandb. Disabling wandb.")
    pass


def get_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-e", "--exp", nargs="+", type=int, required=True, help="Experiment id")
    parser.add_argument("-o", "--output-dir", type=str, default=ROOT_DIR/OUTPUT_FOLDER_NAME, help="Output directory")
    parser.add_argument("-nowb", "--no-wandb", action="store_true", help="Disable weights and biases")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser


def training_loop(
    model,
    optimizer,
    dl_dict: dict,
    config: dict,
    scheduler=None,
    device: str = DEVICE,
    wandb_flag: bool = False,
    output_dir: Path = None
):
    best_accuracy = 0.
    for n_epoch in tqdm(range(config[NB_EPOCHS])):
        current_metrics = {TRAIN: 0., VALIDATION: 0., LR: optimizer.param_groups[0]['lr'],
                           ACCURACY: 0.,
                           PRECISION: 0.,
                           RECALL: 0.,
                           F1_SCORE: 0.,
                           IOU: 0.
                           }
        for phase in [TRAIN, VALIDATION]:
            if phase == TRAIN:
                model.train()
            else:
                model.eval()
            for x, y in tqdm(dl_dict[phase], desc=f"{phase} - Epoch {n_epoch}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == TRAIN):
                    y_pred = model(x)
                    loss = compute_loss(y_pred, y, mode=config.get(LOSS, LOSS_BCE))
                    if torch.isnan(loss):
                        print(f"Loss is NaN at epoch {n_epoch} and phase {phase}!")
                        continue
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()
                current_metrics[phase] += loss.item()
                if phase == VALIDATION:
                    metrics_on_batch = compute_metrics(y_pred, y)
                    for k, v in metrics_on_batch.items():
                        current_metrics[k] += v

            current_metrics[phase] /= (len(dl_dict[phase]))
            if phase == VALIDATION:
                for k, v in metrics_on_batch.items():
                    current_metrics[k] /= (len(dl_dict[phase]))
                    try:
                        current_metrics[k] = current_metrics[k].item()
                    except Exception as e:
                        # Sometimes the metrics are already a float (like 0), so we just pass
                        # print(k, current_metrics[k], e)
                        pass
        print(
            f"{phase}: Epoch {n_epoch} - Loss: {current_metrics[phase]:.3e} " +
            f"Accuracy: {current_metrics[ACCURACY]:.3%}",
            f"Precision: {current_metrics[PRECISION]:.3%} ",
            f"Recall: {current_metrics[RECALL]:.3%} ",
            f"Dice coefficient: {current_metrics[F1_SCORE]:.3%} ",
            f"IoU: {current_metrics[IOU]:.3%}",)
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(current_metrics[VALIDATION])
        if output_dir is not None:
            with open(output_dir/f"metrics_{n_epoch}.json", "w") as f:
                json.dump(current_metrics, f)
        if wandb_flag:
            wandb.log(current_metrics)
        if best_accuracy < current_metrics[ACCURACY]:
            best_accuracy = current_metrics[ACCURACY]
            if output_dir is not None:
                print("new best model saved!")
                torch.save(model.state_dict(), output_dir/"best_model.pt")
    if output_dir is not None:
        torch.save(model.cpu().state_dict(), output_dir/"last_model.pt")
    return model


def train(config: dict, output_dir: Path, device: str = DEVICE, wandb_flag: bool = False):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Training experiment {config[ID]} on device {device}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir/"config.json", "w") as f:
        json.dump(config, f)
    model, optimizer, dl_dict = get_training_content(config, device=device)
    model.to(device)
    if wandb_flag:
        import wandb
        wandb.init(
            project=WANDBSPACE,
            entity="balthazarneveu",
            name=config[NAME],
            # tags=["debug"],
            # tags=["base"],
            tags=["dice_fix", "synthetic"],
            config=config
        )
    scheduler = None
    if config.get(SCHEDULER, False):
        scheduler_config = config[SCHEDULER_CONFIGURATION]
        if config[SCHEDULER] == REDUCELRONPLATEAU:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True, **scheduler_config)
        else:
            raise NameError(f"Scheduler {config[SCHEDULER]} not implemented")
    model = training_loop(model, optimizer, dl_dict, config, scheduler=scheduler, device=device,
                          wandb_flag=wandb_flag, output_dir=output_dir)

    if wandb_flag:
        wandb.finish()


def train_main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    if not WANDB_AVAILABLE:
        args.no_wandb = True
    device = "cpu" if args.cpu else DEVICE
    for exp in args.exp:
        config = get_experiment_config(exp)
        print(config)
        output_dir = Path(args.output_dir)/config[NAME]
        logging.info(f"Training experiment {config[ID]} on device {device}...")
        train(config, device=device, output_dir=output_dir, wandb_flag=not args.no_wandb)


if __name__ == "__main__":
    train_main(sys.argv[1:])
