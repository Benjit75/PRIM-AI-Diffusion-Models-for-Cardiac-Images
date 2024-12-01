import gzip
import os
from typing import Optional, Union
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
                    if image_suffix == '_gt':
                        image_data[f"{image_type}{image_suffix}"] = nib.load(extracted_image_path).get_fdata().astype(np.uint8)
                    else:
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
                    if group_names is None or patient_data['group'] in group_names:  # Check if the patient is in a group to extract
                        for image_type, image_data in patient_data.items():  # Iterate over the image types
                            if isinstance(image_data, dict):  # Check if the image data is a dictionary
                                if image_types is None or image_type in image_types:  # Check if the image type should be extracted
                                    for image_name, image in image_data.items():  # Iterate over the images
                                        if image_names is None or image_name in image_names:
                                            images.append(image)
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

    def filter(self, data_name: Optional[str] = None, groups: Optional[list[str]] = None,
               ids: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Filter the DataFrame based on the provided parameters.
        :param data_name: name of the data (ex: 'train', 'test')
        :param groups: groups to filter (ex: 'NOR', 'MINF', 'DCM', 'HCM', 'RV')
        :param ids: ids to filter (ex: 'patient001', 'patient002', ...)
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

    def display_examples(self, image_type: str='image_data', data_name: Optional[str]=None,
                         groups: Optional[list[str]]=None, image_names: Optional[list[str]]=None,
                         ids: Optional[list[str]]=None, sort_by: Optional[tuple[str]]=None,
                         nb_examples: Optional[int]=None, per_combination: bool=False,
                         format_sep: Optional[tuple[str]] = None, format_categories: Optional[tuple[str]] = None) -> None:
        """
        Display examples based on the provided parameters.
        :param image_type: types of the images to display (ex: 'image_data', 'image_interest_part_data')
        :param data_name: name of the data (ex: 'train', 'test')
        :param groups: groups to filter (ex: 'NOR', 'MINF', 'DCM', 'HCM', 'RV')
        :param image_names: names of the images to extract, or None to extract all (ex: 'ED', 'ES', 'ED_gt', 'ES_gt')
        :param ids: ids to filter (ex: 'patient001', 'patient002', ...)
        :param sort_by: optional tuple of columns to sort by (ex: ('data_name', 'group', 'id'))
        :param nb_examples: number of examples to display
        :param per_combination: whether to sample nb_examples per combination of sort_by columns or globally
        :param format_sep: separator to display between sort_by columns
        :param format_categories: format to display for each sort_by column
        """
        df = self.filter(data_name, groups, ids)

        if image_names is None:
            image_names = ['ED', 'ES', 'ED_gt', 'ES_gt']

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
            self.display_images(row['data_name'], row['id'], image_type=image_type, image_names=image_names)

    def print_metadata(self, data_name: str, id_example: str, indentation: str) -> str:
        """
        Print metadata for a specific example.
        :param data_name: name of the data
        :param id_example: id of the example
        :param indentation: indentation to use
        :return: The formatted metadata
        """
        data = self.data_loader.data[data_name]
        metadata = {
            "ID": id_example,
            "height": f"{data[id_example]['height']:.1f}cm",
            "weight": f"{data[id_example]['weight']:.1f}kg",
            "group": data[id_example]['group'],
            "nb_frames": data[id_example]['nb_frames']
        }
        string_metadata = (indentation + "ID: {ID}, height: {height}, weight: {weight}, group: {group}, nb_frames: {nb_frames}"
                           .format(**metadata))
        print(string_metadata)

        return string_metadata

    def display_images(self, data_name: str, id_example: str, image_type: str, image_names: list[str]) -> None:
        """
        Display images for a specific example.
        :param data_name: name of the data (ex: 'train', 'test')
        :param id_example: id of the example (ex: 'patient001', 'patient002', ...)
        :param image_type: types of the images to display (ex: 'image_data', 'image_resized_data')
        :param image_names: names of the images to extract (ex: 'ED', 'ES', 'ED_gt', 'ES_gt')
        """
        data = self.data_loader.data[data_name]
        fig, axs = plt.subplots(1, len(image_names))
        fig.set_size_inches(10, 10)
        for i, im_name in enumerate(image_names):
            image = data[id_example][image_type][im_name]
            image = image[..., image.shape[-1]//2]
            if image.ndim == 3: # channels image c x h x w
                # transform to h x w x c
                image = np.moveaxis(image, 0, -1)
                axs[i].imshow(image)
            else: # single channel image h x w
                axs[i].imshow(image, cmap='gray')
            axs[i].set_title(f"{im_name}")
        plt.show()

    def display_data_arborescence(self, data_name: str, start_level: int = 0, start_prefix: str = "",
                                  max_keys: Union[int,dict[int, int]] = None, max_depth: int = None) -> str:
        """
        Display the data arborescence.
        :param data_name: name of the data root dictionary
        :param start_level: level of the data dictionary to start from
        :param start_prefix: prefix to start with
        :param max_keys: maximum number of keys to display per level (int or dict with level as key)
        :param max_depth: maximum depth to display
        :return: The formatted data arborescence
        """
        output = []

        def display_data_arborescence_recursive(data: dict, level, prefix):
            nonlocal max_keys, max_depth
            keys = list(data.keys())
            for i, key in enumerate(keys):
                if (isinstance(max_keys, dict) and level in max_keys and i >= max_keys[level]) or (isinstance(max_keys, int) and i >= max_keys):
                    output.append(prefix + "├── ...")
                    break
                output.append(prefix + "├── " + key)
                if isinstance(data[key], dict) and (max_depth is None or level < max_depth):
                    display_data_arborescence_recursive(
                        data[key],
                        level=level + 1,
                        prefix=prefix + "│\t",
                    )

        output.append(start_prefix + data_name)
        display_data_arborescence_recursive(self.data_loader.data, start_level, start_prefix)
        return "\n".join(output)


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

    def crop_and_resize(self,
                        target_shape: tuple[int, int]=(256, 256),
                        padding: float=0.1,
                        image_names: Optional[list[str]]=None,
                        link_gt_to_data: bool=True,
                        keep_3d_consistency: bool=True,
                        create_channels_from_gt: bool=False,
                        output_key: str='image_transformed_data',
                        erase_previous_output: bool=True) -> dict:
        """
        Crop the images to the interesting part and resize them to a target shape.
        :param target_shape: Target shape to resize the images to.
        :param padding: Padding to add to the images before resizing
        :param image_names: List of image names to transform (e.g., ['ED', 'ES', 'ED_gt', 'ES_gt']).
        :param link_gt_to_data: Whether to link the crop parameters of _gt images to their corresponding images.
        :param keep_3d_consistency: Whether to perform the same crop for all layers of 3D images.
        :param create_channels_from_gt: Whether to create channels for the ground truth images.
        :param output_key: The key to store the transformed images in the data dictionary.
        :param erase_previous_output: Whether to erase the previous output if it exists.
        :return: The dictionary of data_loader.data with added output_key.
        """
        if image_names is None:
            image_names = ['ED', 'ES', 'ED_gt', 'ES_gt']
        for dataset_key, dataset in self.data_loader.data.items():
            for patient, patient_data in tqdm(dataset.items(), desc=f"Transforming images in '{dataset_key}'"):
                if erase_previous_output or output_key not in patient_data:
                    images_transformed = {}
                else:
                    images_transformed = patient_data[output_key]

                for image_name in image_names:
                    image = patient_data['image_data'][image_name]
                    process_image_gt = False
                    if link_gt_to_data and '_gt' in image_name:
                        continue  # Skip _gt images if linking is enabled
                    elif link_gt_to_data and f"{image_name}_gt" in image_names:
                        image_gt_name = f"{image_name}_gt"
                        if image_gt_name not in patient_data['image_data']:
                            raise ValueError(f"Image '{image_gt_name}' not found in the data")
                        image_gt = patient_data['image_data'][image_gt_name]
                        if image.shape != image_gt.shape:
                            raise ValueError(f"Image and image_gt shapes do not match for '{image_name}' and '{image_gt_name}'")
                        process_image_gt = True

                    if keep_3d_consistency:
                        non_zero_indices = np.nonzero(image)
                        if len(non_zero_indices[0]) == 0:
                            resized_image = np.zeros(target_shape, dtype=image.dtype)
                            if process_image_gt:
                                resized_image_gt = np.zeros(target_shape, dtype=image_gt.dtype)
                        else:
                            min_dim = [np.min(indices) for indices in non_zero_indices]
                            max_dim = [np.max(indices) for indices in non_zero_indices]
                            slices = tuple(slice(min_dim[dim], max_dim[dim] + 1) for dim in range(image.ndim))
                            cropped_image = image[slices]

                            # Pad the cropped image to a square shape
                            height, width = cropped_image.shape[:2]
                            max_dim = max(height, width)
                            pad_height = (max_dim - height) // 2
                            pad_width = (max_dim - width) // 2
                            padded_image = np.pad(cropped_image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                                                  mode='constant')

                            # Resize the padded image to the target shape
                            pad = int(target_shape[0]*padding)
                            interpolation = cv2.INTER_NEAREST if '_gt' in image_name else cv2.INTER_CUBIC if target_shape[0] - 2*pad > padded_image.shape[0] else cv2.INTER_AREA
                            resized_image = np.zeros((target_shape[0], target_shape[1], padded_image.shape[2]),
                                                    dtype=padded_image.dtype)
                            for i in range(padded_image.shape[2]):
                                resized_image[pad:target_shape[0]-pad, pad:target_shape[1]-pad, i] = (
                                    cv2.resize(padded_image[:, :, i],
                                               (target_shape[0] - 2*pad, target_shape[1] - 2*pad),
                                               interpolation=interpolation))

                            # Process the _gt image if necessary
                            if process_image_gt:
                                cropped_image_gt = image_gt[slices]
                                padded_image_gt = np.pad(cropped_image_gt,
                                                         ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                                                         mode='constant')
                                resized_image_gt = np.zeros(
                                    (target_shape[0], target_shape[1], padded_image_gt.shape[2]),
                                    dtype=padded_image_gt.dtype)
                                for i in range(padded_image_gt.shape[2]):
                                    resized_image_gt[pad:target_shape[0]-pad, pad:target_shape[1]-pad, i] = (
                                        cv2.resize(padded_image_gt[:, :, i],
                                                   (target_shape[0] - 2*pad, target_shape[1] - 2*pad),
                                                   interpolation=cv2.INTER_NEAREST))
                    else:
                        resized_layers = []
                        resized_layers_gt = []
                        for i in range(image.shape[2]):
                            layer = image[:, :, i]
                            non_zero_indices = np.nonzero(layer)
                            if len(non_zero_indices[0]) == 0:
                                resized_layer = np.zeros(target_shape, dtype=layer.dtype)
                                resized_layers.append(resized_layer)
                                if process_image_gt:
                                    resized_layer_gt = np.zeros(target_shape, dtype=layer.dtype)
                                    resized_layers_gt.append(resized_layer_gt)
                            else :
                                min_height = np.min(non_zero_indices[0])
                                max_height = np.max(non_zero_indices[0])
                                min_width = np.min(non_zero_indices[1])
                                max_width = np.max(non_zero_indices[1])
                                cropped_layer = layer[min_height:max_height + 1, min_width:max_width + 1]

                                # Calculate padding to make the cropped layer square
                                height, width = cropped_layer.shape
                                max_dim = max(height, width)
                                pad_height = (max_dim - height) // 2
                                pad_width = (max_dim - width) // 2

                                # Pad the cropped layer to a square shape
                                padded_layer = np.pad(cropped_layer, ((pad_height, pad_height), (pad_width, pad_width)),
                                                      mode='constant')

                                # Resize the padded layer to the target shape
                                pad = int(target_shape[0]*padding)
                                interpolation_layer = cv2.INTER_NEAREST if '_gt' in image_name else cv2.INTER_CUBIC if target_shape[0] - 2*pad > padded_layer.shape[0] else cv2.INTER_AREA
                                resized_layer = np.pad(
                                    cv2.resize(padded_layer, (target_shape[0] - 2*pad, target_shape[1] - 2*pad),
                                               interpolation=interpolation_layer),
                                    ((pad, pad), (pad, pad)), mode='constant')

                                resized_layers.append(resized_layer)

                                # Process the _gt image if necessary
                                if process_image_gt:
                                    layer_gt = patient_data['image_data'][image_gt_name][:, :, i]
                                    cropped_layer_gt = layer_gt[min_height:max_height + 1, min_width:max_width + 1]
                                    padded_layer_gt = np.pad(cropped_layer_gt,
                                                             ((pad_height, pad_height), (pad_width, pad_width)),
                                                             mode='constant')
                                    resized_layer_gt = np.pad(
                                        cv2.resize(padded_layer_gt, (target_shape[0] - 2*pad, target_shape[1] - 2*pad),
                                                   interpolation=cv2.INTER_NEAREST),
                                        ((pad, pad), (pad, pad)), mode='constant')
                                    resized_layers_gt.append(resized_layer_gt)

                        resized_image = np.stack(resized_layers, axis=2)
                        if process_image_gt:
                            resized_image_gt = np.stack(resized_layers_gt, axis=2)

                    if '_gt' in image_name and create_channels_from_gt:
                        resized_image = self.create_channels_from_gt(resized_image)
                    images_transformed[image_name] = resized_image
                    if process_image_gt:
                        if create_channels_from_gt:
                            images_transformed[image_gt_name] = self.create_channels_from_gt(resized_image_gt)
                        else:
                            images_transformed[image_gt_name] = resized_image_gt

                patient_data[output_key] = images_transformed
        return self.data_loader.data

    @staticmethod
    def create_channels_from_gt(image_gt: np.ndarray, channels: int=3) -> np.ndarray:
        """
        Create channels from 3D ground truth image.
        :param image_gt: 3D ground truth image h x w x d.
        :param channels: Number of channels to create.
        :return: Images with channels created from the ground truth image c x h x w x d.
        """
        img_channel = np.zeros((channels, *image_gt.shape), dtype=np.float32)
        for i in range(channels):
            img_channel[i] = (image_gt == i + 1).astype(np.float32)
        return img_channel

    @staticmethod
    def rotate_images(angle: float, images: list[np.ndarray], has_channels:bool=False) -> list[np.ndarray]:
        """
        Rotate images by a given angle.
        :param angle: Angle to rotate the images by.
        :param images: List of 3D images to rotate.
        :param has_channels: Whether the images have channels.
        :return: List of rotated images.
        """
        rotated_images = []
        for image in images:
            if not has_channels:
                if image.ndim == 3:
                    shape = image.shape
                    M = cv2.getRotationMatrix2D((shape[1] / 2, shape[0] / 2), angle, 1)
                    rotated_image = np.stack([cv2.warpAffine(image[:, :, i], M, (shape[1], shape[0])) for i in range(shape[2])], axis=2)
                elif image.ndim == 2:
                    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
                    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                else:
                    raise ValueError("Images must be 2D or 3D: h x w (x d)")
                rotated_images.append(rotated_image)
            else:
                # case of images with channels c x h x w (x d)
                if image.ndim == 4:
                    shape = image.shape
                    M = cv2.getRotationMatrix2D((shape[2] / 2, shape[1] / 2), angle, 1) # rotation around the center
                    rotated_image = np.stack([np.stack([cv2.warpAffine(image[c, :, :, i], M, (shape[2], shape[1])) for i in range(shape[-1])], axis=-1) for c in range(shape[0])], axis=0)
                elif image.ndim == 3:
                    shape = image.shape
                    M = cv2.getRotationMatrix2D((shape[1] / 2, shape[0] / 2), angle, 1)
                    rotated_image = np.stack([cv2.warpAffine(image[c, :, :], M, (shape[1], shape[0])) for c in range(shape[0])], axis=0)
                else:
                    raise ValueError("Images must be 3D or 4D: c x h x w (x d)")
                rotated_images.append(rotated_image)

        return rotated_images

    @staticmethod
    def slice_depth_images(images: list[np.ndarray], create_channel_dim: bool=True) -> list[np.ndarray]:
        """
        Slice depth of images.
        :param images: List of 3D images to slice (c x) h x w x d.
        :param create_channel_dim: Whether to create a channel dimension (if not already present).
        :return: List of sliced images (c x) h x w.
        """
        sliced_images = []
        for image in tqdm(images, desc='Slicing images'):
            shape = image.shape
            for depth in range(shape[-1]):
                slice = image[..., depth]
                if slice.ndim == 2 and create_channel_dim:
                    slice = np.expand_dims(slice, axis=0)
                sliced_images.append(slice)
        return sliced_images