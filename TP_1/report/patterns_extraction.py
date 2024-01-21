import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).parent

MARKER_DATA_PATH = here/".."/"markers_training.pkl"


def get_marker_data(marker_path: Path = MARKER_DATA_PATH) -> dict:
    """Return the markers data read from the pickle file
    """
    with open(marker_path, 'rb') as f:
        dict_markers = pickle.load(f)
    print(", ".join(dict_markers.keys()))
    return dict_markers


def extract_markers_logs(dict_markers: dict, half_neighborhood: int = 64) -> dict:
    """Extract marker patterns from the logs at specified labelled depths

    Args:
        dict_markers (dict): dictionary containing all data and metadata for all wells + labels
        half_neighborhood (int, optional): half neighborhood to extract. Defaults to 64.

    Returns:
        dict: raw patterns, unfiltered and unnormalized
    """
    top_indexes = np.array(dict_markers["top_index"])
    signal = dict_markers["logs"]
    dict_patterns = {}
    total_wells = len(signal)
    for top_type_index, top_type in enumerate(dict_markers["top_names"]):
        dict_patterns[top_type] = np.empty(
            (total_wells, 1+half_neighborhood*2))
        for well_index in range(len(signal)):
            idx_top = top_indexes[well_index][top_type_index]
            if np.isnan(idx_top):
                continue
            idx_top = int(idx_top)
            full_sig = signal[well_index]
            corrupted_indexes = np.where(full_sig < 0)[0]
            for corr_idx in corrupted_indexes:
                full_sig[corr_idx] = 0.
                if corr_idx < full_sig.shape[0]-1:
                    full_sig[corr_idx] = (
                        full_sig[corr_idx-1] + full_sig[corr_idx+1])/2.
                if full_sig[corr_idx] < 0:
                    full_sig[corr_idx] = 0.
            sig = full_sig[idx_top -
                           half_neighborhood: idx_top+half_neighborhood+1]

            dict_patterns[top_type][well_index] = sig
            # sig = (sig - sig.mean())/(sig.std()+1e-10)
            # extracted_depths = depths[well_index][idx_top-neighborhood: idx_top+neighborhood]
    return dict_patterns


def find_template_patterns(dict_patterns: dict, marker_name="CONRAD", plot=True) -> dict:
    patterns = dict_patterns[marker_name].copy()
    average = np.mean(patterns, axis=0)
    sig = patterns.copy()
    n_iter = 3
    for u in range(n_iter):
        threshold = 500  # / (u+1)
        detection_distances = ((sig - average)**2).mean(axis=1)
        # plt.plot(sig.T)
        # plt.plot(average, linewidth=3, color="black")
        # plt.grid()
        # plt.show()
        detection = detection_distances < threshold
        detection_idx = np.where(detection)[0]
        extraction = sig[detection_idx]
        average = np.mean(extraction, axis=0)
        if u == n_iter-1 and plot:
            plt.figure(figsize=(15, 5))
            plt.subplot(121)
            plt.plot(patterns.T, "y-", alpha=0.1)
            plt.plot(extraction.T)
            plt.plot(average, linewidth=3, color="black", label="Average" +
                     f"\nSelected {len(extraction)}/{len(patterns)} signals"
                     + f"\nThreshold: {threshold}")
            plt.ylim(0, 250)  # Warning - we hide certain patterns here.
            plt.title("Extracted patterns")
            plt.grid()
            plt.legend()
            plt.subplot(122)

            # plt.ylim(0, 0.001)
            plt.hist(detection_distances, color="yellow",
                     bins=400, alpha=1., density=True)
            plt.hist(detection_distances[detection_idx], bins=50, density=True)
            plt.plot([threshold, threshold], [0, 0.0001],
                     "r-", label="Threshold")
            plt.xlim(0, threshold*4)
            plt.grid()
            plt.title("Distance histogram")
            plt.suptitle(f"Template extraction {marker_name}")
            plt.show()
    return average


def brute_force_template_matching(dict_markers: dict, templates: dict, sigindex: int = 0, markers=None) -> np.ndarray:
    full_signals = dict_markers["logs"]
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(full_signals[sigindex], "k-", label="Signal", alpha=0.5)
    if markers is None:
        markers = dict_markers["top_names"]
    for marker_index, marker_name in enumerate(markers):
        color = ["r", "g", "b"][marker_index % 3]
        template = templates[marker_name]
        correlation = np.correlate(full_signals[0], template)
        predicted_index = np.argmax(correlation)
        plt.subplot(121)
        indexes = np.arange(len(template)) + predicted_index
        plt.plot(indexes, full_signals[sigindex]
                 [indexes], color, label=f"Matched signal {marker_name}")
        plt.plot(indexes, template, color, label=f"Template {marker_name}")
        plt.subplot(122)
        plt.plot(correlation, color+"-", label=marker_name)
        plt.plot([predicted_index, predicted_index], [
                 0., correlation[predicted_index]],  f"{color}-o", label=f"{predicted_index} Predicted {marker_name}")
    plt.subplot(121)
    plt.legend()
    plt.grid()
    plt.title("Matched signals")
    plt.subplot(122)
    plt.legend()
    plt.grid()
    plt.title("Correlation")
    plt.suptitle("Template matching")
    
    plt.show()


def main():
    dict_markers = get_marker_data()
    dict_patterns = extract_markers_logs(dict_markers)
    template = {}
    for marker_name in dict_markers["top_names"]:
        template[marker_name] = find_template_patterns(
            dict_patterns, marker_name, plot=False)
    dict_markers["top_names"]
    brute_force_template_matching(dict_markers, template, sigindex=0, markers=["CONRAD"])


if __name__ == "__main__":
    main()
