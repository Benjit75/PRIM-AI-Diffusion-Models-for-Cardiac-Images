import numpy as np


def assert_dict_equal(actual, expected, print_diff=False):
    """
    Assert that two dictionaries are equal. If they are not, print the differences if print_diff is True.

    :param actual: The actual dictionary.
    :param expected: The expected dictionary.
    :param print_diff: Boolean flag to print the differences if the dictionaries are not equal.
    """
    def compare_dicts(d1, d2, path=""):
        for key in d1:
            if key not in d2:
                if print_diff:
                    print(f"Missing key in expected: {path + key}")
                return False
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                if not compare_dicts(d1[key], d2[key], path + key + "."):
                    return False
            elif isinstance(d1[key], np.ndarray) and isinstance(d2[key], np.ndarray):
                if not np.array_equal(d1[key], d2[key]):
                    if print_diff:
                        print(f"Difference found in {path + key}")
                        print("Actual:", "\n", d1[key].shape, "\n", d1[key])
                        print("Expected:", "\n", d2[key].shape, "\n", d2[key])
                    return False
            else:
                if d1[key] != d2[key]:
                    if print_diff:
                        print(f"Difference found in {path + key}")
                        print("Actual:", d1[key])
                        print("Expected:", d2[key])
                    return False
        for key in d2:
            if key not in d1:
                if print_diff:
                    print(f"Unexpected key in actual: {path + key}")
                return False
        return True

    if not compare_dicts(actual, expected):
        raise AssertionError("Dictionaries are not equal")