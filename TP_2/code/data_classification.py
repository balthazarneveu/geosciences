import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from itertools import product
import pandas as pd

PRECISION = "precision"
RECALL = "recall"
CONF_MATRIX = "conf_matrix"
CONFIG = "config"
SEQUENCE_LENGTH = "sequence_length"
RATIO_NOISY_LABEL = "ratio_noisy_label"
CLASS_RATIO = "ratio_tight"
CLASSIFIER = "classifier"
DECISION_TREE = "decision_tree"
DECISION_TREE_ON_PCA = "decision_tree_on_pca"
CLASSIFIER_TYPES = [DECISION_TREE, DECISION_TREE_ON_PCA]
TRAIN, EVAL = "train", "eval"
RAW_RESULTS = "raw_results"


def train_and_evaluate(classifier, train_data, train_label, test_data, test_label) -> dict:
    classifier.fit(train_data, train_label)
    full_results_train, light_train = evaluate(classifier, train_data, train_label)
    full_results_test, light_test = evaluate(classifier, test_data, test_label)
    full_results = {
        TRAIN: full_results_train,
        EVAL: full_results_test
    }
    light_results = {
        TRAIN: light_train,
        EVAL: light_test
    }
    return full_results, light_results


def evaluate(classifier, test_data, test_label, debug=False) -> dict:
    predicted_labels = classifier.predict(test_data)

    # Calculate Precision
    precision = precision_score(test_label, predicted_labels, average='binary')  # Change average as per your need

    # Calculate Recall
    recall = recall_score(test_label, predicted_labels, average='binary')  # Change average as per your need

    # Generate the Confusion Matrix
    conf_matrix = confusion_matrix(test_label, predicted_labels,
                                   # normalize='all',
                                   )

    # Print the results
    if debug:
        print(f"Precision: {precision:.2%} | Recall: {recall:.2%} | Confusion Matrix:\n{conf_matrix}")
    full_results = dict(
        precision=precision,
        recall=recall,
        conf_matrix=conf_matrix,
        classifier=classifier,
        predicted_labels=predicted_labels
    )
    light_results = dict(
        precision=precision,
        recall=recall,
        conf_matrix=conf_matrix,
    )
    return full_results, light_results


def get_id(classifier_type=None, sequence_length=None, class_ratio=None, ratio_noisy_label=None):
    id = ""
    if classifier_type is not None:
        id += CLASSIFIER+f"={classifier_type}_"
    if sequence_length is not None:
        id += SEQUENCE_LENGTH+f"={sequence_length:02d}_"
    if class_ratio is not None:
        id += CLASS_RATIO+f"={class_ratio:.2f}_"
    if ratio_noisy_label is not None:
        id += RATIO_NOISY_LABEL+f"={ratio_noisy_label:.2f}_"
    return id[:-1]


if __name__ == '__main__':
    import sys
    from pathlib import Path
    here = Path(__file__).parent
    sys.path.insert(0, here)
    from data_exploration import prepare_whole_dataset, get_data, prepare_data, get_dataset_balance, prepare_augmented_dataset
    data_dict = get_data()
    data, labels = prepare_data(data_dict)
    labels = np.array(labels)
    data = np.array(data)
    raw_train_data, test_data, raw_train_label, test_label = prepare_whole_dataset(data, labels)
    # get_dataset_balance(test_label)
    all_results = {}
    sequence_lengths = [50, 100, 200]
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
            CLASSIFIER_TYPES, sequence_lengths, ratio_noisy_labels, class_ratios):
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
    for iter_noisy_dataset in range(10):
        train_data, train_label = prepare_augmented_dataset(
            raw_train_data, raw_train_label, ratio_noisy_label=ratio_noisy_label, ratio_tight=class_ratio, seed=None)
        # get_dataset_balance(train_label)
        # @TODO: Shuffle
        # classifier = RidgeClassifier()
        for sequence_length in sequence_lengths:  # could iterate here
            train_trim, test_trim = train_data[:, :sequence_length], test_data[:, :sequence_length]
            for classifier_type in CLASSIFIER_TYPES:
                if classifier_type == DECISION_TREE:
                    classifier = DecisionTreeClassifier()
                if classifier_type == DECISION_TREE_ON_PCA:
                    classifier = Pipeline([("pca", PCA(n_components=2)), ("decision_tree", DecisionTreeClassifier())])
                current_id = get_id(classifier_type, sequence_length, class_ratio, ratio_noisy_label)
                full_results, light_results = train_and_evaluate(
                    classifier, train_trim, train_label, test_trim, test_label)
                for mode in [TRAIN, EVAL]:
                    print(current_id)
                    all_results[current_id][RAW_RESULTS][mode] = all_results[current_id][RAW_RESULTS].get(
                        mode, []) + [light_results[mode]]
        # print(light_results['eval']['precision'], light_results['eval']['recall'])
        # print(light_results['eval']['conf_matrix'][1, 0])
        # evaluate(classifier, test_data, test_label)
    # stats = {}
    all_stats = []
    for current_id, value in all_results.items():

        if len(all_results[current_id][RAW_RESULTS][EVAL]) == 0:
            print('SKIP!!!!')
            continue
        else:
            print("PROCESS!", len(all_results[current_id][RAW_RESULTS][EVAL]))
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
            print(mode, all_results[current_id][CONFIG], precision, precision_std, recall, recall_std)
        all_stats.append(stat_dict)
    # SAVE DICT!
    df = pd.DataFrame(all_stats)
    df.to_csv("results.csv")
    # RELOAD DICT AND ANALYZE
    print(all_stats)
    from matplotlib import pyplot as plt
    plt.figure()
    
    for classifier in CLASSIFIER_TYPES:
        df[df[CLASSIFIER] == classifier].plot(x=SEQUENCE_LENGTH, y=EVAL+"_"+PRECISION+"_avg", yerr=EVAL+"_"+PRECISION+"_std", kind='bar')
    # df.plot(x=SEQUENCE_LENGTH, y=EVAL+"_"+PRECISION+"_avg", yerr=EVAL+"_"+PRECISION+"_std", kind='bar')
    
    plt.show()
