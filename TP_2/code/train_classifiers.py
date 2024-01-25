# import sys
# from pathlib import Path
# here = Path(__file__).parent
# sys.path.insert(0, here)

import argparse
from matplotlib import pyplot as plt
from properties import (
    PRECISION, RECALL, ACCURACY,
    FALSE_NEGATIVE, FALSE_POSITIVE,
    CONF_MATRIX, CONFIG, SEQUENCE_LENGTH, RATIO_NOISY_LABEL, CLASS_RATIO, CLASSIFIER,
    DECISION_TREE,
    DECISION_TREE_ON_PCA,
    RIDGE,
    CLASSIFIER_TYPES, TRAIN, EVAL, RAW_RESULTS
)
from data_exploration import prepare_whole_dataset, get_data, prepare_data, prepare_augmented_dataset
import numpy as np
import pandas as pd
from data_classification import train_and_evaluate, get_id
from itertools import product
from pathlib import Path
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def max_trend_feature_extractor(press, labels=None):
    trend = np.linspace(0, 1, 200)[:press.shape[-1]]
    press_minus_trend = press - np.array([trend])
    if labels is not None:
        plt.plot(press_minus_trend[labels == 0, :].T, "r-", alpha=0.1)
        plt.plot(press_minus_trend[labels == 1, :].T, "g-", alpha=0.1)
        plt.show()
    slope = press[:, 20:21]
    max_of_substracted_trend = np.max(press_minus_trend, axis=1, keepdims=True)
    return np.concatenate([slope, max_of_substracted_trend], axis=1)


def plot_dataset(data, labels, title="Full dataset"):
    # plt.plot(data[labels == 0, :].T, "r-", label="tight", alpha=0.2)
    # plt.plot(data[labels == 1, :].T, "g-", label="normal", alpha=0.2)
    # plt.title(f"{title} labelled")
    # plt.show()
    all_feats = max_trend_feature_extractor(data, labels)
    plt.plot(all_feats[labels == 0, 0],
             all_feats[labels == 0, 1], "ro", label="tight")
    plt.plot(all_feats[labels == 1, 0],
             all_feats[labels == 1, 1], "go", label="normal")
    plt.title(f"{title} labelled, hancrafted features")
    plt.grid()
    plt.show()


def main_train(output_path: Path = '__results.csv',
               sequence_lengths=range(20, 200, 10),
               classifier_types=CLASSIFIER_TYPES,
               debug_plots=True,
               ratio_noisy_labels=[0],
               class_ratios=[None],
               ) -> pd.DataFrame:
    data_dict = get_data()
    data, labels = prepare_data(data_dict)
    labels = np.array(labels)
    data = np.array(data)
    # if debug_plots:
    raw_train_data, test_data, raw_train_label, test_label = prepare_whole_dataset(data, labels)
    if debug_plots and False:
        plot_dataset(data, labels, title="Full dataset")
        plot_dataset(raw_train_data, raw_train_label, title="Raw training dataset")
        plot_dataset(test_data, test_label, title="Test dataset")

    all_results = {}

    for classifier_type, sequence_length, ratio_noisy_label, class_ratio in product(
            classifier_types, sequence_lengths, ratio_noisy_labels, class_ratios):
        current_id = get_id(classifier_type, sequence_length, class_ratio, ratio_noisy_label)
        print(current_id)
        all_results[current_id] = {
            CONFIG: {
                SEQUENCE_LENGTH: sequence_length,
                RATIO_NOISY_LABEL: ratio_noisy_label,
                CLASS_RATIO: class_ratio,
                CLASSIFIER: classifier_type
            },
            RAW_RESULTS: {TRAIN: [], EVAL: []}
        }
    # print(all_results)
    for ratio_noisy_label, class_ratio in product(ratio_noisy_labels, class_ratios):
        for iter_noisy_dataset in range(10):
            train_data, train_label = prepare_augmented_dataset(
                raw_train_data.copy(), raw_train_label.copy(),
                ratio_noisy_label=ratio_noisy_label,
                ratio_tight=class_ratio,
                seed=None
            )
            if debug_plots and iter_noisy_dataset == 0 and False:
                plot_dataset(train_data, train_label, title="Augmented training dataset")
            # train_data, train_label = raw_train_data.copy(), raw_train_label.copy()
            # get_dataset_balance(train_label)
            # @TODO: Shuffle
            # classifier = RidgeClassifier()
            for sequence_length in sequence_lengths:  # could iterate here
                train_trim, test_trim = train_data[:, :sequence_length], test_data[:, :sequence_length]
                for classifier_type in classifier_types:

                    feature_extractor = None
                    if classifier_type == DECISION_TREE:
                        classifier = DecisionTreeClassifier()
                    elif classifier_type == DECISION_TREE_ON_PCA:
                        classifier = Pipeline([("pca", PCA(n_components=2)),
                                              ("decision_tree", DecisionTreeClassifier(max_depth=1))])
                    elif classifier_type == RIDGE:
                        classifier = RidgeClassifier()
                    elif classifier_type == "max_of_trend":
                        classifier = DecisionTreeClassifier(max_depth=1)
                        feature_extractor = max_trend_feature_extractor
                    else:
                        raise Exception("Unknown classifier type")

                    if feature_extractor is not None:
                        train_trim_feat, test_trim_feat = feature_extractor(train_trim), feature_extractor(test_trim)
                        # plt.plot(train_trim_feat[train_label == 0, 0],
                        #          train_trim_feat[train_label == 0, 1], "ro", label="tight")
                        # plt.plot(train_trim_feat[train_label == 1, 0],
                        #          train_trim_feat[train_label == 1, 1], "go", label="normal")
                        # plt.title("shuffled data")
                        # plt.show()
                    else:
                        train_trim_feat, test_trim_feat = train_trim, test_trim
                    current_id = get_id(classifier_type, sequence_length, class_ratio, ratio_noisy_label)
                    full_results, light_results, classifier = train_and_evaluate(
                        classifier, train_trim_feat, train_label, test_trim_feat, test_label)
                    if classifier_type == "max_of_trend" and debug_plots and iter_noisy_dataset == 0:
                        # print(classifier)
                        from sklearn import tree
                        print(classifier_type, light_results)
                        tree.plot_tree(classifier)
                        plt.show()
                        plt.plot(train_trim_feat[train_label == 0, 0],
                                 train_trim_feat[train_label == 0, 1], "ro", label="tight")
                        plt.plot(train_trim_feat[train_label == 1, 0],
                                 train_trim_feat[train_label == 1, 1], "go", label="normal")
                        plt.title("shuffled data")
                        plt.show()
                    for mode in [TRAIN, EVAL]:
                        # print(current_id)
                        all_results[current_id][RAW_RESULTS][mode] = all_results[current_id][RAW_RESULTS].get(
                            mode, []) + [light_results[mode]]
        # print(light_results['eval']['precision'], light_results['eval']['recall'])
        # print(light_results['eval']['conf_matrix'][1, 0])
        # evaluate(classifier, test_data, test_label)
    # stats = {}
    all_stats = []
    for current_id, value in all_results.items():
        if len(all_results[current_id][RAW_RESULTS][EVAL]) == 0:
            # print('SKIP!!!!')
            continue
        else:
            # print("PROCESS!", len(all_results[current_id][RAW_RESULTS][EVAL]))
            # print(all_results[current_id][RAW_RESULTS])
            stat_dict = {
                **all_results[current_id][CONFIG]
            }
            stat_dict["id"] = current_id
        for mode in [EVAL, TRAIN]:
            data_serie = all_results[current_id][RAW_RESULTS][mode]
            # print(data_serie)

            df = pd.DataFrame(data_serie)
            # print(df.head())
            accuracy = df.accuracy.mean()
            accuracy_std = df.accuracy.std()
            precision = df.precision.mean()
            precision_std = df.precision.std()
            recall = df.recall.mean()
            recall_std = df.recall.std()
            false_negative = df.false_negative.mean()
            false_negative_std = df.false_negative.std()
            false_positive = df.false_positive.mean()
            false_positive_std = df.false_positive.std()
            stat_dict[mode+"_"+ACCURACY+"_avg"] = accuracy
            stat_dict[mode+"_"+ACCURACY+"_std"] = accuracy_std
            stat_dict[mode+"_"+PRECISION+"_avg"] = precision
            stat_dict[mode+"_"+PRECISION+"_std"] = precision_std
            stat_dict[mode+"_"+RECALL+"_avg"] = recall
            stat_dict[mode+"_"+RECALL+"_std"] = recall_std
            stat_dict[mode+"_"+FALSE_NEGATIVE+"_avg"] = false_negative
            stat_dict[mode+"_"+FALSE_NEGATIVE+"_std"] = false_negative_std
            stat_dict[mode+"_"+FALSE_POSITIVE+"_avg"] = false_positive
            stat_dict[mode+"_"+FALSE_POSITIVE+"_std"] = false_positive_std
        all_stats.append(stat_dict)
    # SAVE DICT!
    df = pd.DataFrame(all_stats)
    df.to_csv(output_path, index=False)
    return df


def analyze_bar_plot(df, classifier_types, value_to_plot=EVAL+"_"+PRECISION, absciss_key=SEQUENCE_LENGTH):
    x = np.arange(len(df[absciss_key].unique()))  # the label locations for each sequence length
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    for i, classifier in enumerate(classifier_types):
        classifier_data = df[df[CLASSIFIER] == classifier]
        classifier_data = classifier_data.sort_values(by=absciss_key)
        offset = width * i
        rects = ax.bar(x + offset, classifier_data[value_to_plot+"_avg"],
                       width, yerr=classifier_data[value_to_plot+"_std"], label=classifier)

        ax.bar_label(rects, padding=3)
    plt.ylim(0, 1)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(value_to_plot+"_avg")
    ax.set_title('Classifier Precision by Sequence Length')
    ax.set_xticks(x + width / 2, df[absciss_key].unique())
    ax.legend()
    plt.grid()

    plt.show()


def plot_curve_with_regard_to_study(df, classifier_types, values_to_analyze=[EVAL+"_"+PRECISION], absciss_key=SEQUENCE_LENGTH):
    # x = np.arange(len(df[SEQUENCE_LENGTH].unique()))  # the label locations for each sequence length

    fig, ax = plt.subplots()

    for i, classifier in enumerate(classifier_types):
        # Extract data for each classifier
        classifier_data = df[df[CLASSIFIER] == classifier]

        # Ensure the data is sorted by SEQUENCE_LENGTH and aligned with x ticks
        classifier_data = classifier_data.sort_values(by=absciss_key)
        for value_to_analyze in values_to_analyze:
            print(classifier_data[absciss_key])
            ax.plot(classifier_data[absciss_key], classifier_data[value_to_analyze+"_avg"],
                    "-o", label=classifier + " " + value_to_analyze)

    plt.ylim(0, 1)
    ax.set_xlabel(absciss_key)
    # ax.set_ylabel(value_to_analyze)
    ax.set_title(f'Classifier performances by {absciss_key}')
    ax.legend()
    plt.grid()
    plt.show()


ALL_CONFIGS = {
    "sequence_lengths_study": dict(
        sequence_lengths=[50, 100, 200],
        classifier_types=[DECISION_TREE_ON_PCA, RIDGE, DECISION_TREE],
        class_ratios=[0.5],
        ratio_noisy_labels=[0],
        picked_study=SEQUENCE_LENGTH
    ),
    "noisy_labels_study": dict(
        sequence_lengths=[200],
        classifier_types=[DECISION_TREE_ON_PCA,],
        class_ratios=[0.5],
        ratio_noisy_labels=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2],
        picked_study=RATIO_NOISY_LABEL
    ),
}

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--force', action="store_true")
    args = argparser.parse_args()
    # chosen_configuration = "noisy_labels_study"
    chosen_configuration = "sequence_lengths_study"
    config = ALL_CONFIGS[chosen_configuration]
    classifier_types = config["classifier_types"]
    sequence_lengths = config["sequence_lengths"]
    class_ratios = config["class_ratios"]
    ratio_noisy_labels = config["ratio_noisy_labels"]
    picked_study = config["picked_study"]
    output_path = Path(f"__results_{chosen_configuration}.csv")
    if not output_path.exists() or args.force:
        df = main_train(
            output_path=output_path,
            classifier_types=classifier_types,
            sequence_lengths=sequence_lengths,
            class_ratios=class_ratios,
            ratio_noisy_labels=ratio_noisy_labels
        )
    else:
        df = pd.read_csv(output_path)
    analyze_bar_plot(
        df, classifier_types,
        value_to_plot=EVAL+"_"+FALSE_NEGATIVE,
        absciss_key=picked_study
    )
    
    plot_curve_with_regard_to_study(df, classifier_types, values_to_analyze=[
        # EVAL+"_"+PRECISION,
        # TRAIN+"_"+PRECISION,
        TRAIN+"_"+ACCURACY,
        EVAL+"_"+ACCURACY,
        TRAIN+"_"+FALSE_NEGATIVE,
        EVAL+"_"+FALSE_NEGATIVE,

        # EVAL+"_"+RECALL
    ],
        absciss_key=picked_study
    )
    analyze_bar_plot(
        df, classifier_types,
        value_to_plot=EVAL+"_"+ACCURACY,
        absciss_key=picked_study
    )
    
