import torch
import pandas as pd
from tqdm import tqdm
from shared import TEST
from pathlib import Path
from shared import ACCURACY, PRECISION, RECALL, F1_SCORE, IOU, VALIDATION
from metrics import compute_metrics


def evaluate_test_mode(model, dl_dict, save_path: Path = None):
    labeled_dict = {}
    dl = dl_dict[TEST]
    for img, img_names in tqdm(dl):
        with torch.no_grad():
            img_names
            output = model(img)
            output = (torch.sigmoid(output) > 0.5).cpu().numpy()*1
            for i, name in enumerate(img_names):
                labeled_dict.update({name: output[i, ...].flatten()})
    if save_path is not None:
        # save_path.write_text(
        pd.DataFrame(labeled_dict, dtype='int').T.to_csv(save_path)
        # )
    return labeled_dict


def evaluate_model(model, dl_dict, phase=VALIDATION):
    current_dataloader = dl_dict[phase]
    current_metrics = {
        ACCURACY: 0.,
        PRECISION: 0.,
        RECALL: 0.,
        F1_SCORE: 0.,
        IOU: 0.
    }
    for img, label in tqdm(current_dataloader):
        with torch.no_grad():
            output = model(img)
            metrics_on_batch = compute_metrics(output, label)
            for k, v in metrics_on_batch.items():
                current_metrics[k] += v
    for k, v in metrics_on_batch.items():
        current_metrics[k] /= (len(current_dataloader))
        current_metrics[k] = current_metrics[k].item()
    print(f"Metrics on {phase} set")
    print(current_metrics)
