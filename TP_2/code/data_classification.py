import numpy as np

from sklearn.metrics import precision_score, recall_score, confusion_matrix
from itertools import product
import pandas as pd
from properties import (
    PRECISION, RECALL, CONF_MATRIX, CONFIG, SEQUENCE_LENGTH, RATIO_NOISY_LABEL, CLASS_RATIO, CLASSIFIER, DECISION_TREE, DECISION_TREE_ON_PCA, CLASSIFIER_TYPES, TRAIN, EVAL, RAW_RESULTS
)


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
