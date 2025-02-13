import ast

import pandas as pd


def test_data_displayer_create_dataframe(data_displayer, data_displayer_create_dataframe_expected_output_path):
    with open(data_displayer_create_dataframe_expected_output_path, 'rb') as f:
        expected_output = pd.read_csv(f, sep=';')

    actual_output = data_displayer.create_dataframe()

    assert pd.DataFrame.equals(actual_output, expected_output)


def test_data_displayer_filter(data_displayer, data_displayer_filter_input_path, data_displayer_filter_expected_output_path):
    with open(data_displayer_filter_input_path, 'r') as f:
        filter_dict = ast.literal_eval(f.read())

    with open(data_displayer_filter_expected_output_path, 'rb') as f:
        expected_output = pd.read_csv(f, sep=';')

    actual_output = data_displayer.filter(**filter_dict)

    assert pd.DataFrame.equals(actual_output, expected_output)


def test_data_displayer_print_metadata(data_displayer, data_displayer_print_metadata_input_path, data_displayer_print_metadata_expected_output_path):
    with open(data_displayer_print_metadata_input_path, 'r') as f:
        inputs_dict = f.read().splitlines()
        inputs_dict = [ast.literal_eval(d) for d in inputs_dict]
    actual_output = [data_displayer.print_metadata(**d) for d in inputs_dict]

    with open(data_displayer_print_metadata_expected_output_path, 'r') as f:
        expected_outputs = f.read().splitlines()

    assert len(actual_output) == len(expected_outputs)
    assert actual_output == expected_outputs


def test_data_displayer_display_data_arborescence(data_displayer, data_displayer_display_data_arborescence_input_path,
                                                  data_displayer_display_data_arborescence_expected_output_path):
    with open(data_displayer_display_data_arborescence_input_path, 'r') as f:
        input_dict = ast.literal_eval(f.read())
    print()
    actual_output = data_displayer.display_data_arborescence(**input_dict)

    with open(data_displayer_display_data_arborescence_expected_output_path, 'r', encoding='utf-8') as f:
        expected_outputs = f.read()
    assert actual_output == expected_outputs
