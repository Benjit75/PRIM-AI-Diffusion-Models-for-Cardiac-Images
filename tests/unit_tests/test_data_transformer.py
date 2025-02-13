import ast

import cv2
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


#noinspection DuplicatedCode
def test_data_transformer_crop_and_resize(data_transformer):
    data_to_load = {
        'test': {
            'patient101': {
                'image_data': {
                    'ED': np.pad(np.ones((4, 4, 2), dtype=float), ((3, 3), (4, 4), (0, 0)), mode='constant'),
                    'ED_gt': np.pad(np.ones((4, 4, 2), dtype=float), ((3, 3), (4, 4), (0, 0)), mode='constant'),
                    'ES': np.pad(np.ones((5, 5, 3), dtype=float), ((5, 5), (3, 3), (0, 0)), mode='constant'),
                    'ES_gt': np.pad(np.ones((5, 5, 3), dtype=float), ((5, 5), (3, 3), (0, 0)), mode='constant')
                },
                'group': 'NOR'
            },
            'patient102': {
                'image_data': {
                    'ED': np.pad(np.ones((3, 3, 1), dtype=float), ((3, 3), (3, 3), (0, 0)), mode='constant'),
                    'ED_gt': np.pad(np.ones((3, 3, 1), dtype=float), ((3, 3), (3, 3), (0, 0)), mode='constant'),
                    'ES': np.pad(np.ones((4, 4, 2), dtype=float), ((4, 4), (4, 4), (0, 0)), mode='constant'),
                    'ES_gt': np.pad(np.ones((4, 4, 2), dtype=float), ((4, 4), (4, 4), (0, 0)), mode='constant')
                },
                'group': 'MINF'
            }
        }
    }
    data_transformer.data_loader.data = data_to_load

    target_shape = (10, 10)
    image_names = ['ED', 'ED_gt']
    actual_output = data_transformer.crop_and_resize(target_shape, image_names=image_names, link_gt_to_data=True,
                                                     keep_3d_consistency=True, create_channels_from_gt=False)

    expected_output = {
        'test': {
            'patient101': {
                'image_data': {
                    'ED': np.pad(np.ones((4, 4, 2), dtype=float), ((3, 3), (4, 4), (0, 0)), mode='constant'),
                    'ED_gt': np.pad(np.ones((4, 4, 2), dtype=float), ((3, 3), (4, 4), (0, 0)), mode='constant'),
                    'ES': np.pad(np.ones((5, 5, 3), dtype=float), ((5, 5), (3, 3), (0, 0)), mode='constant'),
                    'ES_gt': np.pad(np.ones((5, 5, 3), dtype=float), ((5, 5), (3, 3), (0, 0)), mode='constant')
                },
                'group': 'NOR',
                'image_transformed_data': {
                    'ED': np.stack([np.pad(cv2.resize(np.ones((4, 4), dtype=float), (8, 8), interpolation=cv2.INTER_CUBIC),
                                           ((1, 1), (1, 1)), mode='constant') for _ in range(2)], axis=2),
                    'ED_gt': np.stack([np.pad(cv2.resize(np.ones((4, 4)), (8, 8), interpolation=cv2.INTER_NEAREST),
                                              ((1, 1), (1, 1)), mode='constant') for _ in range(2)], axis=2),
                }
            },
            'patient102': {
                'image_data': {
                    'ED': np.pad(np.ones((3, 3, 1), dtype=float), ((3, 3), (3, 3), (0, 0)), mode='constant'),
                    'ED_gt': np.pad(np.ones((3, 3, 1), dtype=float), ((3, 3), (3, 3), (0, 0)), mode='constant'),
                    'ES': np.pad(np.ones((4, 4, 2), dtype=float), ((4, 4), (4, 4), (0, 0)), mode='constant'),
                    'ES_gt': np.pad(np.ones((4, 4, 2), dtype=float), ((4, 4), (4, 4), (0, 0)), mode='constant')
                },
                'group': 'MINF',
                'image_transformed_data': {
                    'ED': np.stack([np.pad(cv2.resize(np.ones((3, 3), dtype=float), (8, 8), interpolation=cv2.INTER_CUBIC),
                                           ((1, 1), (1, 1)), mode='constant') for _ in range(1)], axis=2),
                    'ED_gt': np.stack([np.pad(cv2.resize(np.ones((3, 3), dtype=float), (8, 8), interpolation=cv2.INTER_NEAREST),
                                              ((1, 1), (1, 1)), mode='constant') for _ in range(1)], axis=2),
                }
            }
        }
    }

    assert_dict_equal(actual_output, expected_output, print_diff=True)


def test_data_transformer_create_channels_from_gt(data_transformer):
    img_gt = np.array([
        [
            [0., 1., 2., 3.],
            [1., 3., 0., 0.],
            [2., 0., 0., 1.]
        ],
        [
            [0., 0., 1., 2.],
            [0., 0., 0., 0.],
            [1., 2., 0., 3.]
        ]
    ])

    actual_output = data_transformer.create_channels_from_gt(img_gt)

    expected_output = np.array([
        [
            [
                [0., 1., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 1.]
            ],
            [
                [0., 0., 1., 0.],
                [0., 0., 0., 0.],
                [1., 0., 0., 0.]
            ]
        ],
        [
            [
                [0., 0., 1., 0.],
                [0., 0., 0., 0.],
                [1., 0., 0., 0.]
            ],
            [
                [0., 0., 0., 1.],
                [0., 0., 0., 0.],
                [0., 1., 0., 0.]
            ]
        ],
        [
            [
                [0., 0., 0., 1.],
                [0., 1., 0., 0.],
                [0., 0., 0., 0.]
            ],
            [
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 1.]
            ]
        ]
    ])

    assert np.array_equal(actual_output, expected_output)