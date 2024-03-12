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


def evaluate_model(model, dl_dict, phase=VALIDATION, detailed_metrics_flag=False):
    current_dataloader = dl_dict[phase]
    current_metrics = {
        ACCURACY: 0.,
        PRECISION: 0.,
        RECALL: 0.,
        F1_SCORE: 0.,
        IOU: 0.
    }
    total_elements = 0.
    image_index = 0
    detailed_metrics = []
    for img, label in tqdm(current_dataloader):
        with torch.no_grad():
            output = model(img)
            metrics_on_batch = compute_metrics(output, label, reduce="none")
            if detailed_metrics_flag:
                for element_idx in range(img.shape[0]):
                    current_detail = {k: v[element_idx].item() for k, v in metrics_on_batch.items()}
                    current_name = current_dataloader.dataset.path_list[image_index][0].name
                    current_detail["name"] = current_name
                    current_detail["well"] = int(current_name.split("_")[1])
                    detailed_metrics.append(current_detail)
                    image_index += 1
            for k, v in metrics_on_batch.items():
                current_metrics[k] += v.sum()
            total_elements += img.shape[0]

    for k, v in metrics_on_batch.items():
        current_metrics[k] /= total_elements
        current_metrics[k] = current_metrics[k].item()
    print(f"Metrics on {phase} set")
    print(current_metrics)
    return current_metrics, detailed_metrics
