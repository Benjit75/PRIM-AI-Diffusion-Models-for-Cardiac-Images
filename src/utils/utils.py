import datetime
from enum import Enum
from typing import Optional, Callable, Any, Tuple

import itertools
import numpy as np
import matplotlib.pyplot as plt


class VerboseLevel(Enum):
    NONE = 0
    TQDM = 1
    PRINT = 2
    DISPLAY = 3

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

def format_duration(duration):
    td = datetime.timedelta(seconds=duration)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    microseconds = td.microseconds
    seconds += microseconds / 1000000

    parts = []
    if td.days > 0:
        parts.append(f'{td.days}d')
    if hours > 0:
        parts.append(f'{hours}h')
    if minutes > 0:
        parts.append(f'{minutes}min')
    if seconds > 1:
        parts.append(f'{seconds:.2f}s')
    elif not parts:
        parts.append(f'{microseconds/1000:.2f}ms')

    return ' '.join(parts)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', rotation=0, ha='right')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    print(f'Categorization Accuracy : {np.trace(cm) / np.sum(cm):.2%}')
    return np.trace(cm) / np.sum(cm)


def min_max_scaling(image: np.ndarray, lower_lim: float, upper_lim: float):
    """Min Max scaler of an image between lower_lim and upper_lim"""
    assert lower_lim < upper_lim, ("Lower lim should be strictly lower than upper lim. Get lower_lim={lower_lim}, "
                                   "upper_lim={upper_lim}.")

    return (upper_lim - lower_lim) * (image - image.min()) / (image.max() - image.min()) + lower_lim


def find_divisors(n: int):
    """Find all divisors of n, in the form of a list of tuples (i, n // i)"""
    divisors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append((i, n // i))
    return divisors


def assert_all_same_values_list(list_of_elements: Any, element_name: Tuple[str, str]=('element', 's'),
                                value_compute_fun: Optional[Callable[[Any], Any]] = None,
                                value_name: Tuple[str, str]=('value', 's')) -> None:
    """Assert that all elements in the list have the same value
    :param list_of_elements: list of elements to check
    :param element_name: name of the element to display in the error message, singular and plural
    :param value_compute_fun: function to compute the value of the element, if None, the element is used as value
    :param value_name: name of the value to display in the error message, singular and plural
    """
    values_list = [value_compute_fun(e) for e in list_of_elements] if value_compute_fun is not None else list_of_elements
    different_values = np.where(np.array([v != values_list[0] for v in values_list]))[0]
    assert len(different_values) == 0, \
        (f"All {element_name[0]}{element_name[1]} should have the same {value_name[0]}."
         f"\n\tGot different {value_name[0]}{value_name[1] if len(different_values) > 1 else ''}"
         f" for {element_name[0]}{element_name[1] if len(different_values) > 1 else ''}"
         f" {', '.join([str(i) for i in different_values[:-1]])}"
         f"{' and ' if len(different_values) > 1 else ''}{different_values[-1]}"
         f" (got {value_name[0]}{value_name[1] if len(different_values) > 1 else ''}"
         f" {', '.join([str(values_list[i]) for i in different_values[:-1]])}"
         f"{' and ' if len(different_values) > 1 else ''}{values_list[different_values[-1]]}"
         f" while all {value_name[0]}{value_name[1]} should be {values_list[0]}).")


def assert_or_create_grid_size(desired_size: int, given_size: Optional[Tuple[int, int]]=None) -> Tuple[int, int]:
    """Assert that the given size is a tuple of integers that is a divisor of the desired size or return the tuple of
    integers that is the closest to a square shape of the desired size.
    """
    if given_size is None:
        return find_divisors(desired_size)[-1]

    # Check if the given shape is correct, that is given_shape[0] * given_shape[1] = desired_size and given_shape is a tuple of integers
    assert given_size[0] * given_size[1] == desired_size, f"Shape {given_size[0]} x {given_size[1]} does not match the desired size {desired_size}."
    assert given_size[0] % 1 == given_size[0] and given_size[1] % 1 == given_size[1], f"Given values {given_size} are not integers."
    return given_size
