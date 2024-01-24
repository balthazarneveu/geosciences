# import sys
# from pathlib import Path
# here = Path(__file__).parent
# sys.path.insert(0, here)

import argparse
from matplotlib import pyplot as plt
from properties import (
    PRECISION, RECALL, CONF_MATRIX, CONFIG, SEQUENCE_LENGTH, RATIO_NOISY_LABEL, CLASS_RATIO, CLASSIFIER,
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


def main_train(output_path: Path = '__results.csv', sequence_lengths=range(20, 200, 10), classifier_types=CLASSIFIER_TYPES) -> pd.DataFrame:
    data_dict = get_data()
    data, labels = prepare_data(data_dict)
    labels = np.array(labels)
    data = np.array(data)
    raw_train_data, test_data, raw_train_label, test_label = prepare_whole_dataset(data, labels)
    # get_dataset_balance(test_label)
    all_results = {}
    # sequence_lengths = [50, 100, 200]
    ratio_noisy_labels = [0.2]
    class_ratios = [0.4]
    # all_results = {
    #     "id=...": {
    #         CONFIG: {
    #             SEQUENCE_LENGTH: 50, # 50, 100, 200
    #             RATIO_NOISY_LABEL: 0.2,
    #             CLASS_RATIO: 0.4,
    #             CLASSIFIER: DECISION_TREE # 3 types
    #         }
    #         RAW_RESULTS : [{run1}, {run2}, {run3}, {run4}, {run5}]
    #         RESULTS: {AVERAGE: {}, STD: {}}
    #     }
    # }
    ratio_noisy_label = 0.2
    class_ratio = 0.4
    for classifier_type, sequence_length, class_ratio, ratio_noisy_label in product(
            classifier_types, sequence_lengths, ratio_noisy_labels, class_ratios):
        current_id = get_id(classifier_type, sequence_length, class_ratio, ratio_noisy_label)
        # print(current_id)
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
    for iter_noisy_dataset in range(10):
        train_data, train_label = prepare_augmented_dataset(
            raw_train_data.copy(), raw_train_label.copy(), ratio_noisy_label=ratio_noisy_label, ratio_tight=class_ratio, seed=None)
        # get_dataset_balance(train_label)
        # @TODO: Shuffle
        # classifier = RidgeClassifier()
        for sequence_length in sequence_lengths:  # could iterate here
            train_trim, test_trim = train_data[:, :sequence_length], test_data[:, :sequence_length]
            for classifier_type in classifier_types:
                if classifier_type == DECISION_TREE:
                    classifier = DecisionTreeClassifier()
                elif classifier_type == DECISION_TREE_ON_PCA:
                    classifier = Pipeline([("pca", PCA(n_components=2)), ("decision_tree", DecisionTreeClassifier())])
                elif classifier_type == RIDGE:
                    classifier = RidgeClassifier()
                else:
                    raise Exception("Unknown classifier type")
                current_id = get_id(classifier_type, sequence_length, class_ratio, ratio_noisy_label)
                full_results, light_results = train_and_evaluate(
                    classifier, train_trim, train_label, test_trim, test_label)
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
            precision = df.precision.mean()
            precision_std = df.precision.std()
            recall = df.recall.mean()
            recall_std = df.recall.std()
            stat_dict[mode+"_"+PRECISION+"_avg"] = precision
            stat_dict[mode+"_"+PRECISION+"_std"] = precision_std
            stat_dict[mode+"_"+RECALL+"_avg"] = recall
            stat_dict[mode+"_"+RECALL+"_std"] = recall_std
            # print(mode, all_results[current_id][CONFIG], precision, precision_std, recall, recall_std)
        all_stats.append(stat_dict)
    # SAVE DICT!
    df = pd.DataFrame(all_stats)
    df.to_csv(output_path, index=False)
    return df


def analyze_bar_plot(df, value_to_plot=EVAL+"_"+PRECISION):
    x = np.arange(len(df[SEQUENCE_LENGTH].unique()))  # the label locations for each sequence length
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    for i, classifier in enumerate(CLASSIFIER_TYPES):
        # Extract data for each classifier
        classifier_data = df[df[CLASSIFIER] == classifier]

        # Ensure the data is sorted by SEQUENCE_LENGTH and aligned with x ticks
        classifier_data = classifier_data.sort_values(by=SEQUENCE_LENGTH)

        # Calculate offset for grouped bar chart
        offset = width * i

        # Create bars for this classifier
        rects = ax.bar(x + offset, classifier_data[value_to_plot+"_avg"],
                       width, yerr=classifier_data[value_to_plot+"_std"], label=classifier)

        # Add labels to the bars
        ax.bar_label(rects, padding=3)
    plt.ylim(0, 1)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(value_to_plot+"_avg")
    ax.set_title('Classifier Precision by Sequence Length')
    ax.set_xticks(x + width / 2, df[SEQUENCE_LENGTH].unique())
    ax.legend()

    # Optionally, set the y-axis limit if needed
    # ax.set_ylim(0, max_value)

    plt.show()


def plot_curve_with_regard_to_trim(df, values_to_analyze=[EVAL+"_"+PRECISION]):
    # x = np.arange(len(df[SEQUENCE_LENGTH].unique()))  # the label locations for each sequence length

    fig, ax = plt.subplots()

    for i, classifier in enumerate(CLASSIFIER_TYPES):
        # Extract data for each classifier
        classifier_data = df[df[CLASSIFIER] == classifier]

        # Ensure the data is sorted by SEQUENCE_LENGTH and aligned with x ticks
        classifier_data = classifier_data.sort_values(by=SEQUENCE_LENGTH)
        for value_to_analyze in values_to_analyze:
            ax.plot(classifier_data[SEQUENCE_LENGTH], classifier_data[value_to_analyze+"_avg"],
                    label=classifier + " " + value_to_analyze)

    plt.ylim(0, 1)
    ax.set_xlabel(SEQUENCE_LENGTH)
    # ax.set_ylabel(value_to_analyze)
    ax.set_title('Classifier Precision by Sequence Length')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--force', action="store_true")
    args = argparser.parse_args()
    output_path = Path("__results.csv")
    if not output_path.exists() or args.force:
        df = main_train(
            output_path=output_path,
            # sequence_lengths=range(20, 200, 10),
            sequence_lengths=[8, 50, 200]
        )
    else:
        df = pd.read_csv(output_path)
    plot_curve_with_regard_to_trim(df, values_to_analyze=[
        EVAL+"_"+PRECISION,
        TRAIN+"_"+PRECISION,
        # EVAL+"_"+RECALL
    ]
    )
    analyze_bar_plot(df)
