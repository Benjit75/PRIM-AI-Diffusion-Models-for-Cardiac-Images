import numpy as np
from tqdm.notebook import tqdm

from src.data_manager.datamanager import DataLoader, DataDisplayer, DataTransformer


class ModelPreprocessor:
    def __init__(self, data_loader: DataLoader, group_map: dict[str, str]):
        self.data_loader = data_loader
        self.data_displayer = DataDisplayer(data_loader, group_map)
        self.data_transformer = DataTransformer(data_loader)

    def crop_and_rescale_images(self, target_shape: tuple[int, int], padding: float, image_names: list[str],
                                link_gt_to_data: bool, keep_3d_consistency: bool, create_channels_from_gt: bool,
                                output_key: str, verbose: bool) -> list[np.ndarray]:
        """
        Crop all images to the interesting region and rescale them to the target shape with padding
        :param target_shape: Desired shape of the output images
        :param padding: Padding ratio to apply to the images
        :param image_names: Names of the images to crop and rescale
        :param link_gt_to_data: Use the same transformation for the ground truth segmentation as for the input images
        :param keep_3d_consistency: Preserve 3D consistency of the images i.e. crop and pad all slices in the same way
        :param create_channels_from_gt: Create channels from the ground truth segmentation i.e. one channel per class
        :param output_key: Key to use for storing the output images in the data_loader
        :param verbose: Print information about the cropping and rescaling
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
        if verbose:
            print(
                self.data_displayer.display_data_arborescence(
                    data_name='data_loader.data',
                    max_keys=7
                )
            )

        return self.data_loader.extract_specific_images(
            image_types=[output_key],
            image_names=image_names,
            verbose=verbose
        )

    def data_augmentation(self, images: list[np.ndarray], max_angle: float, nb_rotations: int, verbose: bool) -> list[np.ndarray]:
        """
        Apply data augmentation to the images (rotation)
        :param images: Images to augment
        :param max_angle: Maximum angle of rotation
        :param nb_rotations: Number of rotations to apply
        :param verbose: Print information about the augmentation
        :return: Augmented images
        """
        rotated_images = []
        has_channels = images[0].shape[0] > 3
        for angle in tqdm(np.linspace(-max_angle, max_angle, nb_rotations), disable=not verbose, desc='Rotating images'):
            if angle == 0:
                rotated_images.append(images)
            else:
                rotated_images.append(
                    self.data_transformer.rotate_images(angle=angle, images=images, has_channels=has_channels)
                )
        # Flatten the list of lists
        rotated_images = [image for sublist in rotated_images for image in sublist]
        if verbose:
            print(f'Number of images after rotation: {len(rotated_images)}')

        return rotated_images

    def preprocess_data(self, target_shape: tuple[int, int], padding: float, image_names: list[str],
                        link_gt_to_data: bool, keep_3d_consistency: bool, create_channels_from_gt: bool,
                        rescale_output_key: str, max_angle: float, nb_rotations: int, verbose: bool) -> list[np.ndarray]:
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

        return augmented_images