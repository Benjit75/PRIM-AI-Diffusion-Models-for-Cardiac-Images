import gzip
import os
from typing import Optional
import shutil

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


class DataLoader:
    def __init__(self, root_folder: str):
        """
        Initialize the DataLoader with a root folder.

        Parameters:
        root_folder (str): The root directory where data is stored.
        """
        self.root_folder = root_folder
        self.data = {}

    @staticmethod
    def extract_gz(file_path: str) -> str:
        """
        Extract a .gz file.

        Parameters:
        file_path (str): The path to the .gz file.

        Returns:
        str: The path to the extracted file.
        """
        if file_path.endswith('.gz'):
            output_path = file_path[:-3]  # Remove the .gz extension
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return output_path
        return file_path

    @staticmethod
    def read_metadata(file_path: str) -> dict:
        """
        Read metadata from a file.

        Parameters:
        file_path (str): The path to the metadata file.

        Returns:
        dict: A dictionary containing the metadata.
        """
        metadata = {}
        with open(file_path, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if value.isdigit():
                        metadata[key] = int(value)
                    else:
                        try:
                            metadata[key] = float(value)
                        except ValueError:
                            metadata[key] = value
        return metadata

    def load_data(self, sub_path: str, name: Optional[str] = None, store: Optional[bool] = False) -> dict:
        """
        Load data from a subpath and optionally store it.

        Parameters:
        sub_path (str): The subpath to load data from.
        name (str, optional): The name to store the data under. Defaults to None.
        store (bool, optional): Whether to store the data. Defaults to False.

        Returns:
        dict: A dictionary containing the loaded data.
        """
        folder_path = os.path.join(self.root_folder, sub_path)
        ids = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        data = {}

        for id_subject in tqdm(ids, desc=f"Loading data in '{folder_path}'"):
            metadata_path = os.path.join(folder_path, f"{id_subject}/Info.cfg")
            metadata = self.read_metadata(metadata_path)

            image_data = {}
            for image_type in ['ED', 'ES']:
                for image_suffix in ['', '_gt']:
                    image_path = os.path.join(
                        folder_path,
                        f"{id_subject}/{id_subject}_frame{metadata[image_type]:02}{image_suffix}.nii.gz"
                    )
                    extracted_image_path = self.extract_gz(image_path)
                    image_data[f"{image_type}{image_suffix}"] = nib.load(extracted_image_path).get_fdata()
            data[id_subject] = {"image_data": {k: v for k, v in image_data.items()},
                                "height": metadata["Height"], "weight": metadata["Weight"],
                                "group": metadata["Group"],
                                "nb_frames": metadata["NbFrame"],
                                }

        if store:
            key = name if name else sub_path
            self.data[key] = data

        return data

    def reset_data(self) -> list[str]:
        """
        Reset the data dictionary.

        Returns:
        list[str]: A list of keys that were in the data dictionary.
        """
        keys = list(self.data.keys())
        self.data.clear()
        return keys

    def delete_data(self, key: str) -> dict:
        """
        Delete a specific key from the data dictionary.

        Parameters:
        key (str): The key to delete.

        Returns:
        dict: The value associated with the deleted key, or None if the key was not found.
        """
        return self.data.pop(key, {})

    def extract_specific_images(self,
                                data_names: Optional[list[str]] = None,
                                group_names: Optional[list[str]] = None,
                                image_types: Optional[list[str]] = None,
                                image_names: Optional[list[str]] = None) -> list[np.ndarray]:
        """
        Extract images from the data, filtered by image names and group names.
        :param data_names: names of the data to extract, or None to extract all (ex: 'train', 'test')
        :param group_names: names of the groups to extract, or None to extract all (ex: 'NOR', 'MINF', 'DCM', 'HCM', 'RV')
        :param image_types: types of the images to extract, or None to extract all (ex: 'image_data', 'image_interest_part_data')
        :param image_names: names of the images to extract, or None to extract all (ex: 'ED', 'ES', 'ED_gt', 'ES_gt')
        :return: A numpy array containing the extracted images, in a flattened list.
        """
        images = []
        for dataset_key, dataset in self.data.items():  # Iterate over the datasets
            if data_names is None or dataset_key in data_names:  # Check if the dataset should be extracted
                for patient, patient_data in tqdm(dataset.items(),
                                                  desc=f"Extracting images in '{dataset_key}'"):  # Iterate over the patients
                    if group_names is None or patient_data[
                        'group'] in group_names:  # Check if the patient is in a group to extract
                        for image_type, image_data in patient_data.items():  # Iterate over the image types
                            if isinstance(image_data, dict):  # Check if the image data is a dictionary
                                if image_types is None or image_type in image_types:  # Check if the image type should be extracted
                                    for image_name, image in image_type.items():  # Iterate over the images
                                        if image_names is None or image_name in image_names:
                                            images.append(patient_data['image_data'][image_name])
        return images


class DataDisplayer:
    def __init__(self, data_loader: DataLoader, group_map: dict):
        """
        Initialize the DataDisplayer with a DataLoader and a group map.
        :param data_loader:
        :param group_map:
        """
        self.data_loader = data_loader
        self.data_df = self.create_dataframe()
        self.group_map = group_map

    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame from the data.
        :return: A DataFrame containing the data.
        """
        data = self.data_loader.data
        records = []
        for data_name, examples in data.items():
            for id_example, example_data in examples.items():
                records.append({
                    'data_name': data_name,
                    'group': example_data['group'],
                    'id': id_example
                })
        return pd.DataFrame(records)

    def filter(self, data_name: Optional[str] = None, groups: Optional[list] = None,
               ids: Optional[list] = None) -> pd.DataFrame:
        """
        Filter the DataFrame based on the provided parameters.
        :param data_name: name of the data
        :param groups: groups to filter
        :param ids: ids to filter
        :return: A filtered DataFrame.
        """
        df = self.data_df
        if data_name:
            df = df[df['data_name'] == data_name]
        if groups:
            df = df[df['group'].isin(groups)]
        if ids:
            df = df[df['id'].isin(ids)]
        return df

    def display_examples(self, data_name: Optional[str] = None, groups: Optional[list] = None,
                         ids: Optional[list] = None, sort_by: Optional[tuple] = None, nb_examples: Optional[int] = None,
                         per_combination: bool = False, format_sep: Optional[tuple[str]] = None,
                         format_categories: Optional[tuple[str]] = None) -> None:
        """
        Display examples based on the provided parameters.
        :param data_name: name of the data
        :param groups: groups to filter
        :param ids: ids to filter
        :param sort_by: optional tuple of columns to sort by
        :param nb_examples: number of examples to display
        :param per_combination: whether to sample nb_examples per combination of sort_by columns or globally
        :param format_sep: separator to display between sort_by columns
        :param format_categories: format to display for each sort_by column
        """
        df = self.filter(data_name, groups, ids)

        if nb_examples is not None:
            if per_combination:
                if sort_by[-1] == 'id':
                    df = df.groupby(list(sort_by[:-1])).apply(lambda x: x.sample(min(nb_examples, len(x)))).reset_index(
                        drop=True).sort_values(list(sort_by))
                else:
                    df = df.groupby(list(sort_by)).apply(lambda x: x.sample(min(nb_examples, len(x)))).reset_index(
                        drop=True).sort_values(list(sort_by))
            else:
                df = df.sample(min(nb_examples, len(df))).sort_values(list(sort_by))

        current_sort_values = [None] * len(sort_by)
        indent = ""
        for _, row in df.iterrows():
            if sort_by is not None and format_sep is not None:
                for i, sort_key in enumerate(sort_by):
                    if row[sort_key] != current_sort_values[i]:
                        current_sort_values[i] = row[sort_key]
                        indent = "\t" * i
                        separator = format_sep[i]
                        print(f"{indent}{separator}")
                        if sort_key == 'group':
                            print(indent + format_categories[i].format(self.group_map[current_sort_values[i]]), end='')
                        else:
                            print(indent + format_categories[i].format(current_sort_values[i]), end='')
            self.print_metadata(row['data_name'], row['id'], indent + "\t")
            self.display_images(row['data_name'], row['id'])

    def print_metadata(self, data_name: str, id_example: str, indentation: str) -> None:
        """
        Print metadata for a specific example.
        :param data_name: name of the data
        :param id_example: id of the example
        :param indentation: indentation to use
        """
        data = self.data_loader.data[data_name]
        metadata = {
            "ID": id_example,
            "height": f"{data[id_example]['height']:.1f}cm",
            "weight": f"{data[id_example]['weight']:.1f}kg",
            "group": data[id_example]['group'],
            "nb_frames": data[id_example]['nb_frames']
        }
        print(
            indentation + "ID: {ID}, height: {height}, weight: {weight}, group: {group}, nb_frames: {nb_frames}".format(
                **metadata))

    def display_images(self, data_name: str, id_example: str) -> None:
        """
        Display images for a specific example.
        :param data_name: name of the data
        :param id_example: id of the example
        """
        data = self.data_loader.data[data_name]
        fig, axs = plt.subplots(1, 4)
        fig.set_size_inches(10, 10)
        for i, im_type in enumerate(['ED', 'ES', 'ED_gt', 'ES_gt']):
            axs[i].imshow(data[id_example]['image_data'][im_type][:, :, 0], cmap='gray')
            axs[i].set_title(f"{im_type}")
        plt.show()

    def display_data_arborescence(self, data_name: str, start_level: int = 0, start_prefix: str = "",
                                  max_keys: int = None, max_depth: int = None) -> None:
        """
        Display the data arborescence.
        :param data_name: name of the data root dictionary
        :param start_level: level of the data dictionary to start from
        :param start_prefix: prefix to start with
        :param max_keys: maximum number of keys to display per level
        :param max_depth: maximum depth to display
        """

        def display_data_arborescence_recursive(data: dict, level, prefix):
            nonlocal max_keys, max_depth
            keys = list(data.keys())
            for i, key in enumerate(keys):
                if max_keys is not None and i >= max_keys:
                    print(prefix + "├── ...")
                    break
                print(prefix + "├── " + key)
                if isinstance(data[key], dict) and (max_depth is None or level < max_depth):
                    display_data_arborescence_recursive(
                        data[key],
                        level=level + 1,
                        prefix=prefix + "│\t",
                    )

        print(start_prefix + data_name)
        display_data_arborescence_recursive(self.data_loader.data, start_level, start_prefix)


class DataTransformer:
    """
    DataTransformer class to transform image data from a DataLoader.
    """

    def __init__(self, data_loader: DataLoader):
        """
        Initialize the DataTransformer with a DataLoader.
        :param data_loader: DataLoader instance where the data is stored.
        """
        self.data_loader = data_loader

    def find_images_max_dim(self,
                            data_names: Optional[list[str]] = None,
                            group_names: Optional[list[str]] = None,
                            image_types: Optional[list[str]] = None,
                            image_names: Optional[list[str]] = None) -> tuple[int, int]:
        """
        Detect max length of images in height and width dimensions in the desired data.
        :param data_names: names of the data to extract, or None to extract all (ex: 'train', 'test')
        :param group_names: names of the groups to extract, or None to extract all (ex: 'NOR', 'MINF', 'DCM', 'HCM', 'RV')
        :param image_types: types of the images to extract, or None to extract all (ex: 'image_data', 'image_interest_part_data')
        :param image_names: names of the images to extract, or None to extract all (ex: 'ED', 'ES', 'ED_gt', 'ES_gt')
        :return: The maximum dimensions (height, width).
        """
        max_height = 0
        max_width = 0

        selected_images = self.data_loader.extract_specific_images(data_names, group_names, image_types, image_names)
        for image in selected_images:
            height, width = image.shape[0], image.shape[1]
            max_height = max(max_height, height)
            max_width = max(max_width, width)

        max_shape = (max_height, max_width)
        return max_shape

    def crop_to_interest_part(self) -> dict:
        """
        Detect the interesting part of the images, that is the part containing the heart.
        Crop the images to the interesting part and pad them to a square-shaped image with unchanged depth.
        :return: the dictionary of data_loader.data with added 'image_interest_part_data' key.
        """
        for dataset_key, dataset in self.data_loader.data.items():
            for patient, patient_data in tqdm(dataset.items(), desc=f"Cropping to interesting part in '{dataset_key}'"):
                images_interest_part = {}
                for image_name, image in patient_data['image_data'].items():
                    # Crop the interesting part of the image, that is the part containing the heart, by detecting the non-zero values
                    non_zero_indices = np.nonzero(image)
                    min_dim = [np.min(indices) for indices in non_zero_indices]
                    max_dim = [np.max(indices) for indices in non_zero_indices]
                    slices = tuple(slice(min_dim[dim], max_dim[dim] + 1) for dim in range(image.ndim))
                    cropped_image = image[slices]

                    # Pad the images to a square-shaped image with unchanged depth, centered on the cropped image
                    padded_image = np.zeros(
                        (max(cropped_image.shape[0:1]), max(cropped_image.shape[0:1]), cropped_image.shape[2]),
                        dtype=cropped_image.dtype)
                    pad_height = (padded_image.shape[0] - cropped_image.shape[0]) // 2
                    pad_width = (padded_image.shape[1] - cropped_image.shape[1]) // 2
                    padded_image[pad_height:pad_height + cropped_image.shape[0],
                    pad_width:pad_width + cropped_image.shape[1], :] = cropped_image

                    images_interest_part[image_name] = padded_image
                patient_data['image_interest_part_data'] = images_interest_part
        return self.data_loader.data

    def standardize_images_to_shape(self, target_shape: tuple[int, int]) -> dict:
        """
        Standardize images by resizing them to a target shape along the height and width dimensions.
        :param target_shape: target shape to resize the images to
        :return: the dictionary of data_loader.data with added 'image_resized_data' key.
        """
        for dataset_key, dataset in self.data_loader.data.items():
            for patient, patient_data in tqdm(dataset.items(), desc=f"Resizing images in '{dataset_key}'"):
                images_resized = {}
                for image_name, image in patient_data['image_interest_part_data'].items():
                    resized_image = np.zeros((target_shape[0], target_shape[1], image.shape[2]), dtype=image.dtype)
                    for i in range(image.shape[2]):
                        resized_image[:, :, i] = cv2.resize(image[:, :, i], target_shape, interpolation=cv2.INTER_CUBIC)
                    images_resized[image_name] = resized_image
                patient_data['image_resized_data'] = images_resized
        return self.data_loader.data
