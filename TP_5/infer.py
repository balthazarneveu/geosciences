import argparse
from typing import Optional
from shared import ROOT_DIR, OUTPUT_FOLDER_NAME, DEVICE, NAME, VALIDATION, TRAIN
import sys
from experiments import get_experiment_config, get_training_content
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


def get_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-e", "--exp", nargs="+", type=int, required=True, help="Experiment id")
    parser.add_argument("-o", "--output-dir", type=str, default=ROOT_DIR/OUTPUT_FOLDER_NAME, help="Models directory")
    parser.add_argument("-m", "--mode", type=str, default=VALIDATION,
                        choices=[VALIDATION, TRAIN], help="Mode for inference")

    return parser


def inference_main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    device = DEVICE
    output_dir = Path(args.output_dir)
    for exp in args.exp:
        config = get_experiment_config(exp)
        inference_dir = output_dir/(config[NAME]+"_inference")
        inference_dir.mkdir(exist_ok=True, parents=True)
        model, _, dl_dict = get_training_content(config, device=DEVICE)
        model.load_state_dict(torch.load(output_dir/config[NAME]/"best_model.pt"))
        model.eval()
        model.to(device)
        mode = args.mode
        dataloader = dl_dict[mode]
        running_index = 0
        for img, label in tqdm(dataloader):
            with torch.no_grad():
                output = model(img)
            predicted_mask = (torch.sigmoid(output) > 0.5).cpu().numpy()
            img = img.cpu().numpy()
            label = label.cpu().numpy()
            for idx in range(img.shape[0]):
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(img[idx, 0], cmap="gray")
                plt.subplot(1, 3, 2)
                plt.imshow(predicted_mask[idx, 0])
                plt.subplot(1, 3, 3)
                plt.imshow(label[idx, 0])
                plt.savefig(inference_dir/f"{mode}_{running_index:06d}.png")
                plt.close()
                running_index += 1
                # plt.show()


if __name__ == "__main__":
    inference_main(sys.argv[1:])
