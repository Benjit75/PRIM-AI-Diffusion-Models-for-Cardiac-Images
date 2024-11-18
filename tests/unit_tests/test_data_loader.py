import ast
import os

import nibabel as nib
import numpy as np
import pandas as pd
import pickle


def assert_dict_equal(d1, d2):
    assert d1.keys() == d2.keys()
    for key in d1:
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            assert_dict_equal(d1[key], d2[key])
        else:
            assert np.array_equal(d1[key], d2[key])


def test_data_loader_read_metadata(data_loader, database_folder, data_loader_read_metadata_input_path, data_loader_read_metadata_expected_output_path):
    with open(data_loader_read_metadata_input_path, 'r') as f:
        metadata_inputs_path = f.read().splitlines()
    with open(data_loader_read_metadata_expected_output_path, 'r') as f:
        expected_outputs = f.read().splitlines()
        expected_outputs = [ast.literal_eval(line) for line in expected_outputs]

    actual_outputs = []
    for metadata_input_path in metadata_inputs_path:
        path = os.path.join(database_folder, metadata_input_path)
        actual_outputs.append(data_loader.read_metadata(path))

    assert actual_outputs == expected_outputs


def test_data_loader_extract_gz(data_loader, database_folder, data_loader_extract_gz_input_path, data_loader_extract_gz_expected_output_path):
    with open(data_loader_extract_gz_input_path, 'r') as f:
        gz_inputs_path = f.read().splitlines()
    with open(data_loader_extract_gz_expected_output_path, 'r') as f:
        expected_outputs = f.read().splitlines()
        expected_outputs = [os.path.join(database_folder, path) for path in expected_outputs]

    actual_outputs = []
    for gz_input_path in gz_inputs_path:
        path = os.path.join(database_folder, gz_input_path)
        actual_output = data_loader.extract_gz(path)
        actual_outputs.append(actual_output)

    assert actual_outputs == expected_outputs


def test_data_loader_load_data(data_loader, data_loader_load_data_input_path, data_loader_load_data_expected_output_path):
    data = pd.read_csv(data_loader_load_data_input_path, sep=';')
    for i, row in data.iterrows():
        name = row['name']
        subpath = row['subpath']
        data_loader.load_data(sub_path=subpath, name=name, store=True)

    with open(data_loader_load_data_expected_output_path, 'rb') as f:
        expected_output_data = pickle.load(f)

    assert_dict_equal(data_loader.data, expected_output_data)


def test_data_loader_extract_specific_images(data_loader_loaded, database_folder, data_loader_extract_specific_images_input_path, data_loader_extract_specific_images_expected_output_path):
    with open(data_loader_extract_specific_images_input_path, 'r') as f:
        extract_inputs = ast.literal_eval(f.read())
    with open(data_loader_extract_specific_images_expected_output_path, 'r') as f:
        expected_outputs = f.read().splitlines()
        expected_outputs = [os.path.join(database_folder, path) for path in expected_outputs]
    expected_images = []
    for path in expected_outputs:
        image = nib.load(path).get_fdata()
        expected_images.append(image)

    actual_images = data_loader_loaded.extract_specific_images(**extract_inputs)

    assert len(actual_images) == len(expected_images)

    assert np.array(
        [np.array_equal(actual_image, expected_image)
         for actual_image, expected_image in zip(actual_images, expected_images)]
    ).all()
