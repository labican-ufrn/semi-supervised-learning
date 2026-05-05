import numpy as np


def has_method(obj: object, method_name: str):
    """
    Check if an object has a callable method with the given name.

    Args:
        obj (object): Callable object.
        method_name (str): The name of the method that you want to check.

    Returns:
        bool: True if the object has the method that match with the name
            False, otherwise.
    """
    attribute = getattr(obj, method_name, None)

    return callable(attribute)


def select_labels(labels: np.ndarray, label_percentage: float) -> np.ndarray:
    """
    Selects the instances that will be left without a label randomly
    based on the distribution of classes (stratified selection) and the
    percentage of labeled instances. The minimum number of labeled
    instances per class is 1 instance to ensure the representation of
    minority classes.

    Args:
        labels (np.ndarray): Labels
        labelled_percentage (float): % of instances that will have a
            label.

    Returns:
        np.ndarray: the array of labels already mapped to the labeled
        and unlabeled instances.
    """
    class_dist = np.bincount(labels)
    min_acceptable = np.trunc(class_dist * label_percentage)
    selected = []

    for lab, cls_dist in enumerate(min_acceptable):
        selected += np.random.choice(
            np.where(labels == lab)[0], int(cls_dist) or 1, replace=False
        ).tolist()

    mask = np.ones(len(labels), bool)
    mask[selected] = 0
    labels[mask] = -1

    return labels
