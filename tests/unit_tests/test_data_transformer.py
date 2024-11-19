import ast

import numpy as np

from tests.utils import assert_dict_equal


def test_data_transformer_find_images_max_dim(data_transformer, data_transformer_find_images_max_dim_input_path, data_transformer_find_images_max_dim_expected_output_path):
    data_to_load = {
        'test': {
            'patient101': {
                'image_data': {
                    'ED': np.zeros((10, 12, 5)),
                    'ED_gt': np.zeros((10, 12, 5)),
                    'ES': np.zeros((15, 11, 7)),
                    'ES_gt': np.zeros((15, 11, 7))
                },
                'group': 'NOR'
            },
            'patient102': {
                'image_data': {
                    'ED': np.zeros((9, 10, 4)),
                    'ED_gt': np.zeros((9, 10, 4)),
                    'ES': np.zeros((13, 12, 6)),
                    'ES_gt': np.zeros((13, 12, 6))
                },
                'group': 'MINF'
            },
            'patient103': {
                'image_data': {
                    'ED': np.zeros((10, 14, 5)),
                    'ED_gt': np.zeros((10, 14, 5)),
                    'ES': np.zeros((15, 13, 7)),
                    'ES_gt': np.zeros((15, 13, 7))
                },
                'group': 'NOR'
            },
            'patient104': {
                'image_data': {
                    'ED': np.zeros((10, 10, 6)),
                    'ED_gt': np.zeros((10, 10, 6)),
                    'ES': np.zeros((14, 13, 8)),
                    'ES_gt': np.zeros((14, 13, 8))
                },
                'group': 'NOR'
            }
        },
        'train': {
            'patient001': {
                'image_data': {
                    'ED': np.zeros((8, 10, 3)),
                    'ED_gt': np.zeros((8, 10, 3)),
                    'ES': np.zeros((13, 11, 8)),
                    'ES_gt': np.zeros((13, 11, 8))
                },
                'group': 'MINF'
            },
            'patient002': {
                'image_data': {
                    'ED': np.zeros((16, 15, 8)),
                    'ED_gt': np.zeros((16, 15, 8)),
                    'ES': np.zeros((13, 15, 7)),
                    'ES_gt': np.zeros((13, 15, 7))
                },
                'group': 'NOR'
            },
            'patient003': {
                'image_data': {
                    'ED': np.zeros((15, 16, 6)),
                    'ED_gt': np.zeros((15, 16, 6)),
                    'ES': np.zeros((14, 8, 9)),
                    'ES_gt': np.zeros((14, 8, 9))
                },
                'group': 'DCM'
            },
            'patient004': {
                'image_data': {
                    'ED': np.zeros((9, 9, 11)),
                    'ED_gt': np.zeros((9, 9, 11)),
                    'ES': np.zeros((12, 13, 5)),
                    'ES_gt': np.zeros((12, 13, 5))
                },
                'group': 'RV'
            }
        }
    }
    data_transformer.data_loader.data = data_to_load

    with open(data_transformer_find_images_max_dim_input_path, 'r') as f:
        input_dicts = f.read().splitlines()
        input_dicts = [ast.literal_eval(d) for d in input_dicts]
    actual_outputs = [data_transformer.find_images_max_dim(**d) for d in input_dicts]

    with open(data_transformer_find_images_max_dim_expected_output_path, 'r') as f:
        expected_outputs = f.read().splitlines()
        expected_outputs = [ast.literal_eval(e) for e in expected_outputs]

    assert actual_outputs == expected_outputs


# noinspection DuplicatedCode
def test_data_transformer_crop_to_interest_part(data_transformer):
    data_to_load = {
        'test': {
            'patient101': {
                'image_data': {
                    'ED': np.pad(np.ones((4, 4, 2)), ((3, 3), (4, 4), (1, 1)), mode='constant'),
                    'ED_gt': np.pad(np.ones((4, 4, 2)), ((3, 3), (4, 4), (1, 1)), mode='constant'),
                    'ES': np.pad(np.ones((5, 5, 3)), ((5, 5), (3, 3), (2, 2)), mode='constant'),
                    'ES_gt': np.pad(np.ones((5, 5, 3)), ((5, 5), (3, 3), (2, 2)), mode='constant')
                },
                'group': 'NOR'
            },
            'patient102': {
                'image_data': {
                    'ED': np.pad(np.ones((3, 3, 1)), ((3, 3), (3, 3), (1, 1)), mode='constant'),
                    'ED_gt': np.pad(np.ones((3, 3, 1)), ((3, 3), (3, 3), (1, 1)), mode='constant'),
                    'ES': np.pad(np.ones((4, 4, 2)), ((4, 4), (4, 4), (2, 2)), mode='constant'),
                    'ES_gt': np.pad(np.ones((4, 4, 2)), ((4, 4), (4, 4), (2, 2)), mode='constant')
                },
                'group': 'MINF'
            }
        }
    }
    data_transformer.data_loader.data = data_to_load

    actual_output = data_transformer.crop_to_interest_part()

    expected_output = {
        'test': {
            'patient101': {
                'image_data': {
                    'ED': np.pad(np.ones((4, 4, 2)), ((3, 3), (4, 4), (1, 1)), mode='constant'),
                    'ED_gt': np.pad(np.ones((4, 4, 2)), ((3, 3), (4, 4), (1, 1)), mode='constant'),
                    'ES': np.pad(np.ones((5, 5, 3)), ((5, 5), (3, 3), (2, 2)), mode='constant'),
                    'ES_gt': np.pad(np.ones((5, 5, 3)), ((5, 5), (3, 3), (2, 2)), mode='constant')
                },
                'group': 'NOR',
                'image_interest_part_data': {
                    'ED': np.ones((4, 4, 2)),
                    'ED_gt': np.ones((4, 4, 2)),
                    'ES': np.ones((5, 5, 3)),
                    'ES_gt': np.ones((5, 5, 3))
                }
            },
            'patient102': {
                'image_data': {
                    'ED': np.pad(np.ones((3, 3, 1)), ((3, 3), (3, 3), (1, 1)), mode='constant'),
                    'ED_gt': np.pad(np.ones((3, 3, 1)), ((3, 3), (3, 3), (1, 1)), mode='constant'),
                    'ES': np.pad(np.ones((4, 4, 2)), ((4, 4), (4, 4), (2, 2)), mode='constant'),
                    'ES_gt': np.pad(np.ones((4, 4, 2)), ((4, 4), (4, 4), (2, 2)), mode='constant')
                },
                'group': 'MINF',
                'image_interest_part_data': {
                    'ED': np.ones((3, 3, 1)),
                    'ED_gt': np.ones((3, 3, 1)),
                    'ES': np.ones((4, 4, 2)),
                    'ES_gt': np.ones((4, 4, 2))
                }
            }
        }
    }

    assert_dict_equal(actual_output, expected_output)