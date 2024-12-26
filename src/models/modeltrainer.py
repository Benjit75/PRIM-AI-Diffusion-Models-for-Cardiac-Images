from locale import normalize

import numpy as np
from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler
from torch import nn

from src.data_manager.datamanager import DataLoader, DataDisplayer, DataTransformer
from src.utils.utils import VerboseLevel


class ModelPreprocessor:
    def __init__(self, data_loader: DataLoader, group_map: dict[str, str]):
        self.data_loader = data_loader
        self.data_displayer = DataDisplayer(data_loader, group_map)
        self.data_transformer = DataTransformer(data_loader)

    def crop_and_rescale_images(self, target_shape: tuple[int, int], padding: float, image_names: list[str],
                                link_gt_to_data: bool, keep_3d_consistency: bool, create_channels_from_gt: bool,
                                output_key: str, verbose: VerboseLevel=VerboseLevel.TQDM) -> list[np.ndarray]:
        """
        Crop all images to the interesting region and rescale them to the target shape with padding
        :param target_shape: Desired shape of the output images
        :param padding: Padding ratio to apply to the images
        :param image_names: Names of the images to crop and rescale
        :param link_gt_to_data: Use the same transformation for the ground truth segmentation as for the input images
        :param keep_3d_consistency: Preserve 3D consistency of the images i.e. crop and pad all slices in the same way
        :param create_channels_from_gt: Create channels from the ground truth segmentation i.e. one channel per class
        :param output_key: Key to use for storing the output images in the data_loader
        :param verbose: Verbosity level to display tqdm progress bar and print information
        :return:
        """
        self.data_transformer.crop_and_resize(
            target_shape=target_shape,
            padding=padding,
            image_names=image_names,
            link_gt_to_data=link_gt_to_data,
            keep_3d_consistency=keep_3d_consistency,
            create_channels_from_gt=create_channels_from_gt,
            output_key=output_key,
            verbose=verbose
        )
        if verbose >= VerboseLevel.PRINT:
            # Display the data arborescence
            print(
                self.data_displayer.display_data_arborescence(
                    data_name='data_loader.data',
                    max_keys=7
                )
            )
        if verbose >= VerboseLevel.DISPLAY:
            # Display some examples for the resized images
            self.data_displayer.display_examples(
                image_type=output_key,
                image_names=image_names,
                nb_examples=1,
                per_combination=True,
                sort_by=('data_name', 'group', 'id'),
                format_sep=('#' * 90, '-' * 60, ''),
                format_categories=('{} data :\n', '{} :', '\n')
            )

        return self.data_loader.extract_specific_images(
            image_types=[output_key],
            image_names=image_names,
            verbose=verbose
        )

    def data_augmentation(self, images: list[np.ndarray], max_angle: float, nb_rotations: int,
                          verbose: VerboseLevel=VerboseLevel.TQDM) -> list[np.ndarray]:
        """
        Apply data augmentation to the images (rotation)
        :param images: Images to augment
        :param max_angle: Maximum angle of rotation
        :param nb_rotations: Number of rotations to apply
        :param verbose: Verbosity level to display tqdm progress bar and print information
        :return: Augmented images
        """
        rotated_images = []
        has_channels = images[0].shape[0] > 3
        for angle in tqdm(np.linspace(-max_angle, max_angle, nb_rotations), disable=verbose < VerboseLevel.TQDM,
                          desc='Rotating images'):
            if angle == 0:
                rotated_images.append(images)
            else:
                rotated_images.append(
                    self.data_transformer.rotate_images(angle=angle, images=images, has_channels=has_channels)
                )
        # Flatten the list of lists
        rotated_images = [image for sublist in rotated_images for image in sublist]
        if verbose >= VerboseLevel.PRINT:
            print(f'Number of images after rotation: {len(rotated_images)}')

        return rotated_images

    def slice_depth_images(self, images: list[np.ndarray], create_channel_dim: bool=True,
                           verbose: VerboseLevel=VerboseLevel.TQDM) -> list[np.ndarray]:
        """
        Slice the depth of the images to create 2D slices
        :param images: Images to slice
        :param create_channel_dim: Whether to create a channel dimension for the images if not present
        :param verbose: Verbosity level to display tqdm progress bar and print information
        :return: Sliced images
        """
        sliced_images = self.data_transformer.slice_depth_images(
            images=images,
            create_channel_dim=create_channel_dim,
            verbose=verbose
        )
        if verbose >= VerboseLevel.PRINT:
            print(f'Number of images after slicing: {len(sliced_images)}')

        return sliced_images

    def one_hot_encode_images(self, images: list[np.ndarray], one_hot_encode: bool,
                              verbose: VerboseLevel=VerboseLevel.TQDM) -> list[np.ndarray]:
        """
        One-hot encode the images
        :param images: Images to one-hot encode
        :param one_hot_encode: Whether to one-hot encode the images or not
        :param verbose: Verbosity level to display tqdm progress bar and print information
        :return: One-hot encoded images if necessary
        """
        if one_hot_encode:
            one_hot_encoded_images = self.data_transformer.one_hot_encode_batch(
                images=images,
                verbose=verbose
            )
            if verbose >= VerboseLevel.PRINT:
                print('One-hot encoding applied')
            return one_hot_encoded_images
        else:
            if verbose >= VerboseLevel.PRINT:
                print('No one-hot encoding applied')
            return images

    @staticmethod
    def normalize_images(images: list[np.ndarray], verbose: VerboseLevel=VerboseLevel.TQDM) -> list[np.ndarray]:
        """
        Normalize the images
        :param images: Images to normalize
        :param verbose: Verbosity level to display tqdm progress bar and print information
        :return: Normalized images
        """
        flat_images = np.concatenate([img.flatten() for img in images])
        scaler = MinMaxScaler(feature_range=(-1., 1.))
        scaler.fit(flat_images.reshape(-1, 1))
        normalized_images = []
        for img in tqdm(images, disable=verbose < VerboseLevel.TQDM, desc='Normalizing images'):
            normalized_images.append(scaler.transform(img.reshape(-1, 1)).reshape(img.shape))
        return normalized_images

    def preprocess_data(self, target_shape: tuple[int, int], padding: float, image_names: list[str],
                        link_gt_to_data: bool, keep_3d_consistency: bool, create_channels_from_gt: bool,
                        rescale_output_key: str, max_angle: float, nb_rotations: int, one_hot_encode: bool=False,
                        verbose: VerboseLevel=VerboseLevel.TQDM) -> list[np.ndarray]:
        """
        Preprocess the data by cropping, rescaling and augmenting the images
        :param target_shape: Target shape for rescaling the images
        :param padding: Padding ratio to apply to the images
        :param image_names: Names of the images to preprocess
        :param link_gt_to_data: Whether to link the ground truth segmentation to the input images
        :param keep_3d_consistency: Whether to preserve 3D consistency of the images i.e. crop and pad all slices in the same way
        :param create_channels_from_gt: Whether to create channels from the ground truth segmentation i.e. one channel per class
        :param rescale_output_key: Key to use for storing the rescaled images in the data_loader
        :param max_angle: Maximum angle of rotation to apply during data augmentation
        :param nb_rotations: Number of rotations to apply during data augmentation
        :param one_hot_encode: Whether to one-hot encode the ground truth segmentation
        :param verbose: Print information about the preprocessing
        :return: Preprocessed images
        """
        rescaled_images = self.crop_and_rescale_images(
            target_shape=target_shape,
            padding=padding,
            image_names=image_names,
            link_gt_to_data=link_gt_to_data,
            keep_3d_consistency=keep_3d_consistency,
            create_channels_from_gt=create_channels_from_gt,
            output_key=rescale_output_key,
            verbose=verbose
        )
        augmented_images = self.data_augmentation(
            images=rescaled_images,
            max_angle=max_angle,
            nb_rotations=nb_rotations,
            verbose=verbose
        )
        sliced_images = self.slice_depth_images(
            images=augmented_images,
            create_channel_dim=create_channels_from_gt,
            verbose=verbose
        )
        one_hot_encoded_images = self.one_hot_encode_images(
            images=sliced_images,
            one_hot_encode=one_hot_encode,
            verbose=verbose
        )

        return one_hot_encoded_images


class ModelTrainer:
    """ Class to train the models """

    def __init__(self, data_set: list[np.ndarray], batch_size: int,
                 model: nn.Module, loss_fn: nn.Module, optimizer: nn.Module):
        self.data_set = data_set
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

