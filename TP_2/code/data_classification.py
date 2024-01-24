import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# First, predict the labels for the test set

classifier = RidgeClassifier()
classifier = DecisionTreeClassifier()


def train_and_evaluate(classifier, train_data, train_label, test_data, test_label) -> dict:
    classifier.fit(train_data, train_label)
    results = {
        "train": evaluate(classifier, train_data, train_label),
        "eval": evaluate(classifier, test_data, test_label),
    }
    return results


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
    return dict(
        precision=precision,
        recall=recall,
        conf_matrix=conf_matrix,
        classifier=classifier,
        predicted_labels=predicted_labels
    )


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
    get_dataset_balance(test_label)
    for iter_noisy_dataset in range(10):
        train_data, train_label = prepare_augmented_dataset(
            raw_train_data, raw_train_label, ratio_noisy_label=0.2, ratio_tight=0.4, seed=None)
        get_dataset_balance(train_label)
        # @TODO: Shuffle
