import torch
import pandas as pd
from tqdm import tqdm
from shared import TEST
from pathlib import Path
from shared import ACCURACY, PRECISION, RECALL, F1_SCORE, IOU, VALIDATION
from metrics import compute_metrics
from matplotlib import pyplot as plt


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


def visualize_performance_per_well(df, chosen_metrics=[F1_SCORE, IOU], title='Metrics per Well'):
    # Group by 'well' and calculate mean
    grouped = df.groupby('well')[chosen_metrics].mean()

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Twin the axes for two different y-axis scales
    ax2 = ax1.twinx()
    axs = [ax1, ax2]
    colors = ['g', 'orange']

    # Plotting
    for idx, chosen_metric in enumerate(chosen_metrics):
        grouped[chosen_metric].plot(kind='bar', ax=axs[idx], color=colors[idx],
                                    position=idx, width=0.4, label=chosen_metric)
    # grouped[F1_SCORE].plot(kind='bar', ax=ax1, position=1, color='blue', width=0.4, label=F1_SCORE)
    # grouped[IOU].plot(kind='bar', ax=ax2, position=0, color='green', width=0.4, label=IOU)

    # Setting the axis labels
    ax1.set_xlabel('Well')
    for idx, chosen_metric in enumerate(chosen_metrics):
        axs[idx].set_ylabel(chosen_metric)
    # ax1.set_ylabel(F1_SCORE, color='blue')
    # ax2.set_ylabel(IOU, color='green')

    # Setting the tick label size
    # ax1.tick_params(axis='y', colors='blue')
    # ax2.tick_params(axis='y', colors='green')

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    for ax in axs:
        # ax.set_ylim(0., 1.)
        ax.set_ylim(0.4, 0.9)

    plt.grid()
    plt.title(title)
    plt.show()


def compare_performance_per_well(df_list, label_list, chosen_metric=F1_SCORE, title='Metrics per Well'):
    # Group by 'well' and calculate mean

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['g', 'orange', "cyan", "red"]
    # Plotting
    for idx, df in enumerate(df_list):
        grouped = df.groupby('well')[chosen_metric].mean()
        grouped.plot(kind='bar', ax=ax1, color=colors[idx],
                     position=idx, width=0.2, label=label_list[idx])
    ax1.set_xlabel('Well')
    ax1.set_ylabel(chosen_metric)
    ax1.set_ylim(0.4, 1.)
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.show()


def get_global_metrics_str(metrics_dict):
    global_metrics_str = ""
    for key, value in metrics_dict.items():
        global_metrics_str += f"{key}: {value:.1%} "
    return global_metrics_str
