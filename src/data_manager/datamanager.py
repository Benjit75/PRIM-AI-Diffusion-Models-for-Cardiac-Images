import gzip
import IPython.display
import os
from typing import Optional, Union, Tuple
import shutil

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont, Image
import torch
from torch import Tensor
from torchvision.utils import make_grid
from tqdm.notebook import tqdm

from utils.utils import VerboseLevel, assert_all_same_values_list, assert_or_create_grid_size


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

    def load_data(self, sub_path: str, name: Optional[str] = None, store: Optional[bool] = False,
                  verbose: VerboseLevel=VerboseLevel.TQDM) -> dict:
        """
        Load data from a subpath and optionally store it.

        Parameters:
        sub_path (str): The subpath to load data from.
        name (str, optional): The name to store the data under. Defaults to None.
        store (bool, optional): Whether to store the data. Defaults to False.
        verbose (VerboseLevel, optional): The verbosity level. Defaults to VerboseLevel.TQDM.

        Returns:
        dict: A dictionary containing the loaded data.
        """
        folder_path = os.path.join(self.root_folder, sub_path)
        ids = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        data = {}

        for id_subject in tqdm(ids, disable=verbose < VerboseLevel.TQDM, desc=f"Loading data in '{folder_path}'"):
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
                    image_data[f"{image_type}{image_suffix}"] = nib.load(extracted_image_path).get_fdata().astype(np.float32)

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
                                image_names: Optional[list[str]] = None,
                                verbose: VerboseLevel=VerboseLevel.TQDM) -> list[np.ndarray]:
        """
        Extract images from the data, filtered by image names and group names.
        :param data_names: names of the data to extract, or None to extract all (ex: 'train', 'test')
        :param group_names: names of the groups to extract, or None to extract all (ex: 'NOR', 'MINF', 'DCM', 'HCM', 'RV')
        :param image_types: types of the images to extract, or None to extract all (ex: 'image_data', 'image_interest_part_data')
        :param image_names: names of the images to extract, or None to extract all (ex: 'ED', 'ES', 'ED_gt', 'ES_gt')
        :param verbose: whether to display the progress bar
        :return: A numpy array containing the extracted images, in a flattened list.
        """
        images = []
        for dataset_key, dataset in self.data.items():  # Iterate over the datasets
            if data_names is None or dataset_key in data_names:  # Check if the dataset should be extracted
                for patient, patient_data in tqdm(dataset.items(), disable=verbose < VerboseLevel.TQDM,
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
                         ids: Optional[list[str]]=None, sort_by: Optional[tuple[str, ...]]=None,
                         nb_examples: Optional[int]=None, per_combination: bool=False,
                         format_sep: Optional[tuple[str, ...]] = None,
                         format_categories: Optional[tuple[str, ...]] = None) -> None:
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
            if len(image.shape) == 3: # channels image c x h x w
                # transform to h x w x c
                image = np.moveaxis(image, 0, -1)
                one_hot_image = DataTransformer.one_hot_encode(image)
                axs[i].imshow(image[:, :, 1:image.shape[2]])
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

    @staticmethod
    def display_batch(batch: torch.Tensor, grid_shape: Optional[Tuple[int, int]]=None, show: bool=True,
                      filename: Optional[str]=None, title: Optional[str]=None, one_hot_encode: bool=False) -> None:
        """ Display a batch of images, with possibility to save the display.
        :param batch: images to display. Must be a tensor of shape b x c x h x w
        :param grid_shape: shape of the grid to display the images, shape row x col and row * col = batch_size
        :param show: whether to display the images or not, for saving only purposes
        :param filename: Optional path to save the display if needed
        :param title: Optional title to display
        :param one_hot_encode: use a argmax encoder along the channel dimension
        """
        if filename is None and not show:
            # Nothing to display or save, skipping
            return

        # Verify that the grid shape is correct or find the nearest square grid shape
        n_row, n_col = assert_or_create_grid_size(desired_size=len(batch), given_size=grid_shape)

        image = np.array(
            torch.permute(make_grid(batch, nrow=n_col, padding=0),(1, 2, 0)).cpu())
        image = (image + np.ones(image.shape)) / 2. # Scale the normalization from [-1, 1] to [0, 1]
        if image.shape[-1] > 3:
            if one_hot_encode:
                image = DataTransformer.one_hot_encode(image)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(image[..., 1:4])
            axes[0].set_title(("" if title is None else f"{title} - ") + "RGB Channels")
            axes[0].axis('off')
            im_bg = axes[1].imshow(image[..., 0], cmap='viridis')
            axes[1].set_title(("" if title is None else f"{title} - ") + "Background Intensity")
            axes[1].axis('off')
            cbar = fig.colorbar(im_bg, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label("Intensity")
            plt.tight_layout()
            if filename:
                fig.savefig(filename)
            if show:
                plt.show()
            else:
                plt.close()
        else:
            plt.imshow(image, cmap='gray')
            if filename:
                plt.savefig(filename)
            if show:
                plt.show()
            else:
                plt.close()

    @staticmethod
    def display_batches_comparisons(batches, grid_shape: Optional[Tuple[int, int]]=None, show: bool=True,
                                    filename: Optional[str]=None, one_hot_encode: bool=False, title: Optional[str]=None,
                                    labels: Optional[list[str]]=None) -> None:
        """
        Display a grid of size (1 row, n columns) with each cell showing a grid of images from one batch.
        :param batches: List of tensors of shape torch.Size([16, 4, 128, 128]).
        :param grid_shape: Tuple of (rows, columns) for the grid corresponding of 1 batch, see display_batch for more details.
        :param show: Whether to display the images or not, for saving only purposes. See display_batch for more details.
        :param filename: Optional path to save the display if needed.
        :param one_hot_encode: Whether to one-hot encode the images for display.
        :param title: Optional title to display.
        :param labels: Optional list of labels to display for each batch.
        """
        if filename is None and not show:
            # Nothing to display or save, skipping
            return

        fig, axes = plt.subplots(1, len(batches), figsize=(18, 6))

        # Verify that all batches have the same shape
        assert_all_same_values_list(
            list_of_elements=batches,
            element_name=('batch', 'es'),
            value_compute_fun=lambda b: b.shape,
            value_name=('shape', 's')
        )

        # Verify that the grid shape is correct or find the nearest square grid shape
        n_row, n_col = assert_or_create_grid_size(desired_size=batches[0].shape[0], given_size=grid_shape)

        for i, batch in enumerate(batches):
            image = np.array(torch.permute(make_grid(batch, nrow=n_col, padding=0), (1, 2, 0)).cpu())
            image = (image + np.ones(image.shape)) / 2. # Scale the normalization from [-1, 1] to [0, 1]

            if image.shape[-1] > 3:
                if one_hot_encode:
                    image = DataTransformer.one_hot_encode(image)
                axes[i].imshow(image[..., 1:4]) # Display RGB channels only
            else:
                axes[i].imshow(image, cmap='gray')

            axes[i].axis('off')
            axes[i].set_title(labels[i] if labels else f"Batch {i+1}")

        plt.tight_layout()
        if title:
            fig.suptitle(title, fontsize=16)
        if filename:
            plt.savefig(filename)
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def make_gif(frame_list: list[Tensor], filename: str, grid_size: Optional[Tuple[int, int]]=None,
                 step: int=5, one_hot_encode: bool=False, verbose: VerboseLevel=VerboseLevel.TQDM) -> None:
        """
        Create a GIF from a list of frames corresponding to a batch of images at different timesteps and save it.
        :param frame_list: List of frames to create the GIF from.
        :param filename: path to save the GIF.
        :param grid_size: Size of the grid to display the images, shape row x col and row * col = batch_size.
        :param step: Step to take between frames.
        :param one_hot_encode: Whether to one-hot encode the images for display.
        :param verbose: Whether to display the gif created.
        """
        # Verify that all frames have the same shape
        assert_all_same_values_list(
            list_of_elements=frame_list,
            element_name=('frame', 's'),
            value_compute_fun=lambda f: f.shape,
            value_name=('shape', 's')
        )

        # Verify that the grid size is correct or find the nearest square grid size
        n_row, n_col = assert_or_create_grid_size(desired_size=len(frame_list[0]), given_size=grid_size)

        frames = [np.array(torch.permute(make_grid(tens_im, nrow=n_col, padding=0), (1,2,0)).cpu()) for tens_im in frame_list]
        frames = [(frame + np.ones(frame.shape))/2. for frame in frames] # Scale the normalization from [-1, 1] to [0, 1]
        frames_to_include = frames[0::step] # Take every step frames
        if len(frames) % step != 0:
            frames_to_include.append(frames[-1]) # Add the last frame if not included
        if frames_to_include[0].shape[-1] > 3:
            if one_hot_encode:
                frames_to_include = [DataTransformer.one_hot_encode(frame) for frame in frames_to_include]
            frames_to_include = [frame[..., 1:4] for frame in frames_to_include] # Display RGB channels only
        frames_pil = [Image.fromarray((255*frame).astype(np.uint8)) for frame in frames_to_include]
        frame_one = frames_pil[0]

        frame_one.save(filename, format="GIF", append_images=frames_pil[1::], save_all=True, duration=10, loop=0)
        if verbose >= VerboseLevel.PRINT:
            print(f"Saved GIF at {filename}.")
        if verbose >= VerboseLevel.DISPLAY:
            IPython.display.display(IPython.display.Image(filename=filename))

    @staticmethod
    def make_gif_comparison(frame_lists: list[list[Tensor]], filename: str, grid_size: Optional[Tuple[int, int]]=None,
                            step: int=5, one_hot_encode: bool=False, padding: int=10, title: Optional[str]=None,
                            subtitles: Optional[str]=None, verbose: VerboseLevel=VerboseLevel.TQDM) -> None:
        """
        Create and save a GIF of grid (1 row, n columns) from a list of several frames list corresponding to a batch of
        images at different timesteps for different sampling and save it.
        :param frame_lists: List of frame lists to create the GIF from.
        :param filename: path to save the GIF.
        :param grid_size: Size of the grid to display the images, shape row x col and row * col = the size of the batch.
        :param step: Step to take between frames.
        :param one_hot_encode: Whether to one-hot encode the images for display.
        :param padding: Padding to add between the images.
        :param title: Optional title to display.
        :param subtitles: Optional subtitles to display.
        :param verbose: Whether to display the progress bar and the gif created.
        """
        # Verify that all list have the same length
        assert_all_same_values_list(
            list_of_elements=frame_lists,
            element_name=('frame list', 's'),
            value_compute_fun=lambda f: len(f),
            value_name=('length', 's')
        )

        # Verify that all frames have the same shape in each list
        assert_same_shape_loop = tqdm(frame_lists, disable=verbose < VerboseLevel.TQDM, desc="Checking frames shapes")
        for frame_list in assert_same_shape_loop:
            assert_all_same_values_list(
                list_of_elements=frame_list,
                element_name=('frame', 's'),
                value_compute_fun=lambda f: f.shape,
                value_name=('shape', 's')
            )

        # Verify that all lists have elements with the same shape
        assert_all_same_values_list(
            list_of_elements=frame_lists,
            element_name=('frame list', 's'),
            value_compute_fun=lambda f: f[0].shape,
            value_name=('shape', 's')
        )

        # Verify that the grid size is correct or find the nearest square grid size
        n_row, n_col = assert_or_create_grid_size(desired_size=len(frame_lists[0][0]), given_size=grid_size)

        num_batches = len(frame_lists)
        num_timesteps = len(frame_lists[0])
        frames = []
        font = ImageFont.load_default()

        # Create the frames for the GIF
        for t in tqdm(range(0, num_timesteps, step), disable=verbose < VerboseLevel.TQDM, desc="Creating frames"):
            row_images = []
            for b in range(num_batches):
                batch = frame_lists[b][t]
                image = np.array(
                    torch.permute(make_grid(batch, nrow=n_col, padding=0), (1, 2, 0)).cpu())
                image = (image + np.ones(image.shape)) / 2.0 # Scale the normalization from [-1, 1] to [0, 1]
                if image.shape[-1] > 3:
                    if one_hot_encode:
                        image = DataTransformer.one_hot_encode(image)
                    image = image[..., 1:4] # Display RGB channels only
                row_images.append(image)

            # Concatenate row images into one frame
            padded_row = np.concatenate(
                [np.pad(img, ((0, 0), (padding, padding), (0, 0)), constant_values=1) if idx != num_batches - 1 else img for
                 idx, img in enumerate(row_images)], axis=1)

            # Convert the row to an image and overlay text
            frame = Image.fromarray((255*padded_row).astype(np.uint8))
            draw = ImageDraw.Draw(frame)

            # Add the title if specified
            if title:
                draw.text((frame.width // 2 - len(title) * 3, 5), title, fill="black", font=font, anchor="mm")

            # Add subtitles for each epoch
            if subtitles:
                column_width = frame.width // num_batches
                for i, subtitle in enumerate(subtitles):
                    x_position = column_width * i + column_width // 2
                    draw.text((x_position, 20), subtitle, fill="black", font=font, anchor="mm")

            frames.append(frame)

        # Save the frames as a GIF
        frame_one = frames[0]
        frame_one.save(filename, format="GIF", append_images=frames[1:], save_all=True, duration=10, loop=0)

        if verbose >= VerboseLevel.PRINT:
            print(f"Saved GIF at {filename}.")
        if verbose >= VerboseLevel.DISPLAY:
            IPython.display.display(IPython.display.Image(filename=filename))


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
                        erase_previous_output: bool=True,
                        verbose: VerboseLevel=VerboseLevel.TQDM) -> dict:
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
        :param verbose: Whether to display the progress bar.
        :return: The dictionary of data_loader.data with added output_key.
        """
        if image_names is None:
            image_names = ['ED', 'ES', 'ED_gt', 'ES_gt']
        for dataset_key, dataset in self.data_loader.data.items():
            for patient, patient_data in tqdm(dataset.items(), disable=verbose < VerboseLevel.TQDM,
                                              desc=f"Transforming images in '{dataset_key}'"):
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
    def create_channels_from_gt(image_gt: np.ndarray, channels: int=4) -> np.ndarray:
        """
        Create channels from 3D ground truth image.
        :param image_gt: 3D ground truth image h x w x d.
        :param channels: Number of channels to create.
        :return: Images with channels created from the ground truth image c x h x w x d.
        """
        img_channel = np.zeros((channels, *image_gt.shape), dtype=np.float32)
        for i in range(channels):
            img_channel[i] = (np.round(image_gt) == i).astype(np.float32)
        return img_channel

    @staticmethod
    def rotate_images(angle: float, images: list[np.ndarray], has_channels: bool = False, epsilon: float = 1e-3) -> list[np.ndarray]:
        """
        Rotate images by a given angle. For images with channels, fills 0-pixels caused by rotation 
        with 1 in the background channel (first channel).
        
        :param angle: Angle to rotate the images by (in degrees).
        :param images: List of 2D, 3D, or 4D images to rotate.
        :param has_channels: Whether the images have channels.
        :param epsilon: 0 used for float comparison
        :return: List of rotated images.
        """
        rotated_images = []
        for image in images:
            if not has_channels:
                if image.shape[0] == 3:  # 3D image (h x w x d)
                    shape = image.shape
                    M = cv2.getRotationMatrix2D((shape[1] / 2, shape[0] / 2), angle, 1)
                    rotated_image = np.stack(
                        [cv2.warpAffine(image[:, :, i], M, (shape[1], shape[0])) for i in range(shape[2])],
                        axis=2,
                    )
                elif image.shape[0] == 2:  # 2D image (h x w)
                    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
                    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                else:
                    raise ValueError("Images must be 2D or 3D: h x w (x d)")
                rotated_images.append(rotated_image)
            else:
                # Case of images with channels c x h x w (x d)
                if image.shape[0] == 4:  # 4D image (c x h x w x d)
                    shape = image.shape
                    M = cv2.getRotationMatrix2D((shape[2] / 2, shape[1] / 2), angle, 1)  # Rotate around the center
                    rotated_image = np.stack(
                        [
                            np.stack(
                                [cv2.warpAffine(image[c, :, :, i], M, (shape[2], shape[1])) for i in range(shape[-1])],
                                axis=-1,
                            )
                            for c in range(shape[0])
                        ],
                        axis=0,
                    )
                elif image.shape[0] == 3:  # 3D image (c x h x w)
                    shape = image.shape
                    M = cv2.getRotationMatrix2D((shape[2] / 2, shape[1] / 2), angle, 1)
                    rotated_image = np.stack(
                        [cv2.warpAffine(image[c, :, :], M, (shape[2], shape[1])) for c in range(shape[0])],
                        axis=0,
                    )
                else:
                    raise ValueError("Images must be 3D or 4D: c x h x w (x d)")
                # Handle zero-fill pixels generated by rotation
                zeros_mask = np.all(rotated_image[1:4, ...] < epsilon, axis=0)  # Check along all RGB channels
                rotated_image[0, zeros_mask] = 1.  # Set the background channel to 1 where mask is True
                rotated_images.append(rotated_image)
    
        return rotated_images
        
    @staticmethod
    def slice_depth_images(images: list[np.ndarray], create_channel_dim: bool=True,
                           verbose: VerboseLevel=VerboseLevel.TQDM) -> list[np.ndarray]:
        """
        Slice depth of images.
        :param images: List of 3D images to slice (c x) h x w x d.
        :param create_channel_dim: Whether to create a channel dimension (if not already present).
        :param verbose: Whether to display the progress bar.
        :return: List of sliced images (c x) h x w.
        """
        sliced_images = []
        for image in tqdm(images, disable=verbose < VerboseLevel.TQDM, desc='Slicing images'):
            shape = image.shape
            for depth in range(shape[-1]):
                slice = image[..., depth]
                if slice.ndim == 2 and create_channel_dim:
                    slice = np.expand_dims(slice, axis=0)
                sliced_images.append(slice)
        return sliced_images

    @staticmethod
    def one_hot_encode(image: np.ndarray) -> np.ndarray:
        """
        One-hot encode the image, that is set to 1 the argmax of channels and others to 0.
        :param image: image to one-hot-encode, shape h x w (x d) x c
        :return: The corresponding one-hot image
        """
        one_hot_image = np.zeros_like(image)
        max_indices = np.argmax(image, axis=-1)
        if len(image.shape) == 3:
            h, w, c = image.shape
            rows, cols, = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            one_hot_image[rows, cols, max_indices] = 1.
        elif len(image.shape) == 4:
            h, w, d, c = image.shape
            rows, cols, depths = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing='ij')
            one_hot_image[rows, cols, depths, max_indices] = 1.  # Set the max index to 1 in the one-hot array
        else:
            raise ValueError("Images must be 3D or 4D: h x w (x d) x c")
        return one_hot_image

    @staticmethod
    def one_hot_encode_batch(images : list[np.ndarray], verbose: VerboseLevel=VerboseLevel.TQDM) -> list[np.ndarray]:
        """
        One-hot-encode a list of images
        :param images: the list of images to one-hot-encode, shape c x h x w
        :param verbose: whether to display the progress bar
        :return: List of one-hot-encoded images
        """
        one_hot_images = []
        for image in tqdm(images, disable=verbose < VerboseLevel.TQDM, desc='One-hot-encoding images'):
            one_hot_images.append(DataTransformer.one_hot_encode(image.transpose(1, 2, 0)).transpose(2, 0, 1))

        return one_hot_images
