import pytest
import os

from data_manager.datamanager import DataLoader, DataDisplayer, DataTransformer

# General paths
@pytest.fixture
def root_test_folder():
    """
    Define the root test folder from the parent of current directory
    :return: the root test folder
    """
    if os.getcwd().split(os.sep)[-1] == "tests":
        return os.getcwd()
    else:
        return os.path.dirname(os.getcwd())

@pytest.fixture
def database_folder(root_test_folder):
    return os.path.join(root_test_folder, "data", "database")

@pytest.fixture
def inputs_folder(root_test_folder):
    return os.path.join(root_test_folder, "unit_tests", "inputs")

@pytest.fixture
def expected_outputs_folder(root_test_folder):
    return os.path.join(root_test_folder, "unit_tests", "expected_outputs")


# Instances
@pytest.fixture
def data_sub_folders():
    return {
        "train": "training",
        "test": "testing",
    }

@pytest.fixture
def group_map():
    return {
        "NOR": "Healthy control",
        "MINF": "Myocardial infarction",
        "DCM": "Dilated cardiomyopathy",
        "HCM": "Hypertrophic cardiomyopathy",
        "RV": "Abnormal right ventricle"
    }

@pytest.fixture
def data_loader(database_folder):
    return DataLoader(database_folder)

@pytest.fixture
def data_loader_loaded(database_folder, data_sub_folders):
    data_loader_loaded = DataLoader(database_folder)
    for key, sub_folder in data_sub_folders.items():
        data_loader_loaded.load_data(sub_path=sub_folder, name=key, store=True)
    return data_loader_loaded

@pytest.fixture
def data_displayer(data_loader_loaded, group_map):
    return DataDisplayer(data_loader=data_loader_loaded, group_map=group_map)

@pytest.fixture
def data_transformer(data_loader_loaded):
    return DataTransformer(data_loader_loaded)


# Unit test paths

## DataLoader
@pytest.fixture
def data_loader_read_metadata_input_path(inputs_folder):
    return os.path.join(inputs_folder, "data_loader_read_metadata.txt")

@pytest.fixture
def data_loader_read_metadata_expected_output_path(expected_outputs_folder):
    return os.path.join(expected_outputs_folder, "data_loader_read_metadata.txt")

@pytest.fixture
def data_loader_extract_gz_input_path(inputs_folder):
    return os.path.join(inputs_folder, "data_loader_extract_gz.txt")

@pytest.fixture
def data_loader_extract_gz_expected_output_path(expected_outputs_folder):
    return os.path.join(expected_outputs_folder, "data_loader_extract_gz.txt")

@pytest.fixture
def data_loader_load_data_input_path(inputs_folder):
    return os.path.join(inputs_folder, "data_loader_load_data.csv")

@pytest.fixture
def data_loader_load_data_expected_output_path(expected_outputs_folder):
    return os.path.join(expected_outputs_folder, "data_loader_load_data.pkl")

@pytest.fixture
def data_loader_extract_specific_images_input_path(inputs_folder):
    return os.path.join(inputs_folder, "data_loader_extract_specific_images.txt")

@pytest.fixture
def data_loader_extract_specific_images_expected_output_path(expected_outputs_folder):
    return os.path.join(expected_outputs_folder, "data_loader_extract_specific_images.txt")


## DataDisplayer
@pytest.fixture
def data_displayer_create_dataframe_expected_output_path(expected_outputs_folder):
    return os.path.join(expected_outputs_folder, "data_displayer_create_dataframe.csv")

@pytest.fixture
def data_displayer_filter_input_path(inputs_folder):
    return os.path.join(inputs_folder, "data_displayer_filter.txt")

@pytest.fixture
def data_displayer_filter_expected_output_path(expected_outputs_folder):
    return os.path.join(expected_outputs_folder, "data_displayer_filter.csv")

@pytest.fixture
def data_displayer_print_metadata_input_path(inputs_folder):
    return os.path.join(inputs_folder, "data_displayer_print_metadata.txt")

@pytest.fixture
def data_displayer_print_metadata_expected_output_path(expected_outputs_folder):
    return os.path.join(expected_outputs_folder, "data_displayer_print_metadata.txt")

@pytest.fixture
def data_displayer_display_data_arborescence_input_path(inputs_folder):
    return os.path.join(inputs_folder, "data_displayer_display_data_arborescence.txt")

@pytest.fixture
def data_displayer_display_data_arborescence_expected_output_path(expected_outputs_folder):
    return os.path.join(expected_outputs_folder, "data_displayer_display_data_arborescence.txt")


# DataTransformer
@pytest.fixture
def data_transformer_find_images_max_dim_input_path(inputs_folder):
    return os.path.join(inputs_folder, "data_transformer_find_images_max_dim.txt")

@pytest.fixture
def data_transformer_find_images_max_dim_expected_output_path(expected_outputs_folder):
    return os.path.join(expected_outputs_folder, "data_transformer_find_images_max_dim.txt")