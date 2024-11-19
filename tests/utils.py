import numpy as np

def assert_dict_equal(d1, d2):
    assert d1.keys() == d2.keys()
    for key in d1:
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            assert_dict_equal(d1[key], d2[key])
        else:
            assert np.array_equal(d1[key], d2[key])