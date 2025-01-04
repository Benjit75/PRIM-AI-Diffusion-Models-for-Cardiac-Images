import warnings
from typing import Tuple, Union, Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, random_split, Subset
from tqdm.notebook import tqdm

from src.data_manager.datamanager import DataLoader, DataDisplayer, DataTransformer
from src.utils.utils import VerboseLevel, min_max_scaling


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


class Diffusion:
    """ Class to define the diffusion algorithm """

    @staticmethod
    def linear_beta_schedule(timesteps: int, beta_start: float=0.0001, beta_end: float=0.02) -> Tensor:
        """
        Linear beta schedule
        :param timesteps: Number of timesteps
        :param beta_start: Initial beta value
        :param beta_end: Final beta value
        :return: Beta values for each timestep
        """
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float=0.008) -> Tensor:
        """
        Cosine beta schedule
        :param timesteps: Number of timesteps
        :param s: Scaling factor
        :return: Beta values for each timestep
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def get_alph_bet(timesteps: int, schedule: Callable[[int], Tensor]) -> dict[
        str, Union[Tensor, float]]:
        """
        Get alpha and beta values for each timestep
        :param timesteps: Number of timesteps
        :param schedule: Beta schedule to use
        :return: Alpha and beta values for each timestep
        """
        betas = schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        const_dict = {
            'betas': betas,
            'sqrt_recip_alphas': sqrt_recip_alphas,
            'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
            'posterior_variance': posterior_variance
        }

        return const_dict

    @staticmethod
    def extract(constants: Union[Tensor, float], batch_t: Tensor, x_shape: tuple[int, int, int]) -> Tensor:
        """
        Extract the image from the batch
        :param constants: Constants for the model
        :param batch_t: Batch of images
        :param x_shape: Shape of the images
        :return: Extracted image
        """
        diffusion_batch_size = batch_t.shape[0]
        out = (constants.gather(-1, batch_t.cpu())
               .reshape(diffusion_batch_size, *((1,) * (len(x_shape) - 1)))
               .to(batch_t.device))
        return out

    @staticmethod
    def q_sample(constants_dict: dict[str, Union[Tensor, float]], batch_x0: Tensor, batch_t: Tensor,
                 noise: Optional[Tensor]=None) -> Tensor:
        """
        Sample from the model
        :param constants_dict: Constants for the model
        :param batch_x0: Batch of images
        :param batch_t: Batch of timesteps
        :param noise: Noise to add to the images
        :return: Sampled images
        """
        if noise is None:
            noise = torch.randn_like(batch_x0)

        sqrt_alpas_cumprod_t = Diffusion.extract(constants_dict['sqrt_alphas_cumprod'], batch_t, batch_x0.shape)
        sqrt_one_minus_alphas_cumprod_t = Diffusion.extract(constants_dict['sqrt_one_minus_alphas_cumprod'],
                                                                 batch_t, batch_x0.shape)

        return sqrt_alpas_cumprod_t * batch_x0 + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, constants_dict: dict[str, Union[Tensor, float]], batch_xt: Tensor, predicted_noise: Tensor,
                 batch_t: Tensor) -> Tensor:
        """
        Sample one step ahead from the model
        :param constants_dict: constants for the model
        :param batch_xt: batch of images at time t
        :param predicted_noise: predicted noise for the images
        :param batch_t: batch of timesteps
        :return: Sampled images one step ahead
        """
        # Extract constants and move to the correct device
        betas_t = self.extract(constants_dict['betas'], batch_t, batch_xt.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(constants_dict['sqrt_one_minus_alphas_cumprod'], batch_t,
                                                       batch_xt.shape)
        sqrt_recip_alphas_t = self.extract(constants_dict['sqrt_recip_alphas'], batch_t, batch_xt.shape)
        posterior_variance_t = self.extract(constants_dict['posterior_variance'], batch_t, batch_xt.shape)

        # Calculate model mean
        model_mean = sqrt_recip_alphas_t * (batch_xt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        # Initialize predicted image
        predicted_image = model_mean.clone()
        t_zero_index = (batch_t == 0)

        # Add noise if t != 0
        noise = torch.randn_like(batch_xt)
        predicted_image[~t_zero_index] += torch.sqrt(posterior_variance_t[~t_zero_index]) * noise[~t_zero_index]

        return predicted_image

    @torch.no_grad()
    def sampling(self, model: nn.Module, shape: Tuple[int, int, int, int], timesteps: int,
                 constants_dict: dict[str, Union[Tensor, float]], device: torch.device,
                 start: Optional[Tensor] = None, verbose: VerboseLevel = VerboseLevel.TQDM) -> list[Tensor]:
        """
        Sampling method to create images from diffusion model
        :param model: trained diffusion model to use
        :param shape: shape of images in the format batch x channels x height x width
        :param timesteps: Number of timesteps
        :param constants_dict: Constants to use for the diffusion process
        :param device: torch device to use for the computation
        :param start: optional tensor of shape 'shape', if none passed, start the diffusion from random noise
        :param verbose: Verbosity to use in order to display tqdm progress bar or not
        :return: list of tensor representing one batch of images sampled during all timesteps
        """
        # Start from pure noise if no start tensor is provided
        batch_xt = torch.randn(shape, device=device) if start is None else start.clone()

        batch_t = torch.full((shape[0],), timesteps, dtype=torch.int64, device=device)

        imgs = []

        for t in tqdm(reversed(range(timesteps)), disable=verbose < VerboseLevel.TQDM, desc='sampling loop time step',
                      total=timesteps):
            batch_t -= 1
            predicted_noise = model(batch_xt, batch_t)
            batch_xt = self.p_sample(constants_dict, batch_xt, predicted_noise, batch_t)
            imgs.append(batch_xt.cpu())

        return imgs

    @torch.no_grad()
    def sampling(self, model: nn.Module, shape: Tuple[int, int, int, int], timesteps: int,
                 constants_dict: dict[str, Union[Tensor, float]], device: torch.device,
                 start: Optional[Tensor]=None, verbose: VerboseLevel=VerboseLevel.TQDM) -> list[Tensor]:
        """
        Sampling method to create images from diffusion model
        :param model: trained diffusion model to use
        :param shape: shape of images in the format batch x channels x height x width
        :param timesteps: Number of timesteps
        :param constants_dict: Constants to use for the diffusion process
        :param device: torch device to use for the computation
        :param start: optional tensor of shape 'shape', if none passed, start the diffusion from random noise
        :param verbose: Verbosity to use in order to display tqdm progress bar or not
        :return: list of tensor representing one batch of images sampled during all timesteps
        """
        # start from pure noise (for each example in the batch)
        if start is None:
            batch_xt = torch.randn(shape, device=device)
        else:
            assert start.shape == shape, "Incorrect value passed as argument for `start`. It should be a tensor of the shape `shape`."
            batch_xt = start.clone()

        batch_t = torch.ones(shape[0]) * timesteps  # create a vector with batch-size time the timestep
        batch_t = batch_t.type(torch.int64).to(device)

        imgs = []

        for t in tqdm(reversed(range(0, timesteps)), disable=verbose<VerboseLevel.TQDM, desc='sampling loop time step',
                      total=timesteps):
            batch_t -= 1
            predicted_noise = model(batch_xt, batch_t)

            batch_xt = self.p_sample(constants_dict, batch_xt, predicted_noise, batch_t)

            imgs.append(batch_xt.cpu())

        return imgs


class DiffusionModelTrainer:
    """ Class to train the model """

    def __init__(self,
                 data_set: list[np.ndarray],
                 val_split: float,
                 batch_size: int,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device
                ):
        self.batch_size = batch_size
        self.train_loader, self.val_loader, self.unused_images = self.split_train_val(data_set, val_split, batch_size)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    @staticmethod
    def split_train_val(data_set: list[np.ndarray], val_split: float, batch_size: int,
                        verbose: VerboseLevel=VerboseLevel.PRINT) -> Tuple[TorchDataLoader, TorchDataLoader, Subset[np.ndarray]]:
        """
        Split data into train val and unused sets, with keeping val_split part of the data to validation set.
        Some images are unused due to the fact that we want a multiple of batch_size in each subset
        :param data_set: dataset of images to split
        :param val_split: part to use for the validation set
        :param batch_size: number of images per batch
        :param verbose: Verbosity level to print info about the splits
        :return: train and val torch data loaders and unused images
        """
        assert 0. <= val_split < 1., "Val_split must be between 0 and 1 (excluded) to cut the dataset."
        assert len(data_set) >= batch_size, "Batch size greater than the data set: cannot create data loaders."
        unused_size = len(data_set) % batch_size
        val_size = batch_size * int((val_split * (len(data_set) - unused_size) // batch_size) // 1)
        train_size = len(data_set) - unused_size - val_size

        assert train_size > 0, "Train set is empty, consider decreasing the val_split or the batch_size"
        if val_size == 0 and val_split > 0.:
            warnings.warn("WARNING: val set is empty, consider increasing the val_split or decreasing the batch_size")

        if verbose >= VerboseLevel.PRINT:
            print(f'{train_size} images ({train_size // batch_size} batchs) for train, '
                  f'{val_size} images ({val_size // batch_size} batchs) for validation, '
                  f'({train_size / len(data_set):.1%}-{val_size / len(data_set):.1%})'
                  f'\n{unused_size} images left unused.')

        train_images, val_images, unused_images = random_split(data_set, [train_size, val_size, unused_size])
        return (TorchDataLoader(dataset=train_images, batch_size=batch_size, shuffle=True),
                TorchDataLoader(dataset=val_images, batch_size=batch_size, shuffle=True),
                unused_images)

    @staticmethod
    def normalize_batch_images(batch: torch.Tensor, verbose: VerboseLevel=VerboseLevel.TQDM):
        """
        Normalize each image individually in the batch between -1 and 1 using a MinMax scaling
        :param batch: images to normalize, shape b x c x h x w
        :param verbose: Verbosity level to display tqdm progress bar
        :return: the normalized images as a batch
        """
        normalized_images = []
        for image in tqdm(batch, disable=verbose < VerboseLevel.TQDM, desc="Normalizing images in batch"):
            normalized_images.append(min_max_scaling(image=image, lower_lim=-1., upper_lim=1.))
        return torch.stack(normalized_images)

    def train(self, epochs: int, timesteps: int, constants_scheduler: Callable[[int], Tensor],
              save_model_path: Optional[str]=None, save_losses_path: Optional[str]=None,
              save_intermediate_models: Optional[dict[str, Union[bool, int]]]=None,
              verbose: VerboseLevel=VerboseLevel.TQDM) -> dict[int, dict[str, float]]:
        """
        Train the model
        :param epochs: Number of epochs to train the model
        :param timesteps: Number of timesteps to use for the model
        :param constants_scheduler: Function to get the constants for the model
        :param save_model_path: Path to save the model, not saved if None
        :param save_losses_path: Path to save the loss graph
        :param save_intermediate_models: Dictionary with keys 'toggle' and 'frequency' to save the model every 'frequency' epochs
        :param verbose: Verbosity level to display tqdm progress bar
        :return: Training history with train and validation losses
        """
        if save_intermediate_models is None:
            save_intermediate_models = {'toggle': False, 'frequency': 1}
        constants_dict = Diffusion.get_alph_bet(timesteps, constants_scheduler)
        timesteps_val = torch.randint(0, timesteps, (self.batch_size,), device=self.device).long()  # fixed time steps for validation

        history = {}
        best_val_loss = float('inf')  # Start with an infinitely high loss
        best_model_state = None  # To save the best model
        best_epoch = 0
        epochs_loop = tqdm(range(epochs), disable=verbose < VerboseLevel.TQDM, desc='Training epochs')
        for epoch in epochs_loop:
            train_loss = self.train_epoch(timesteps=timesteps, epoch=epoch+1, constants_dict=constants_dict, verbose=verbose)
            val_loss = self.val_epoch(epoch=epoch+1, constants_dict=constants_dict, timesteps_val=timesteps_val, verbose=verbose)
            if verbose >= VerboseLevel.PRINT:
                print(f'Epoch {epoch + 1} - Train loss: {train_loss:.5f} - Val loss: {val_loss:.5f}')
            history[epoch + 1] = {'train_loss': train_loss, 'val_loss': val_loss}
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                best_epoch = epoch + 1
            epochs_loop.set_postfix(best_epoch=best_epoch, best_val_loss=best_val_loss)
            if (save_model_path is not None and save_intermediate_models['toggle']
                    and (epoch + 1) % save_intermediate_models['frequency'] == 0):
                torch.save(self.model.state_dict(), save_model_path.format(f'-epoch-{epoch + 1}-'))
        if verbose >= VerboseLevel.PRINT:
            print(f'Best epoch: {best_epoch + 1} - Best val loss: {best_val_loss:.5f}')

        if save_model_path is not None:
            torch.save(best_model_state, save_model_path.format(f'-best-epoch-{best_epoch}-'))

        self.monitor_training(history=history, best_epoch=best_epoch, save_losses_path=save_losses_path, logscale=False,
                              verbose=verbose)
        self.monitor_training(history=history, best_epoch=best_epoch, save_losses_path=save_losses_path, logscale=True,
                              verbose=verbose)
        return history

    def train_epoch(self, timesteps: int, epoch: int, constants_dict: dict[str, Union[Tensor, float]],
                    verbose: VerboseLevel=VerboseLevel.TQDM) -> float:
        """
        Train the model for one epoch
        :param timesteps: Number of timesteps to use for the model
        :param epoch: Current epoch
        :param constants_dict: Constants for the model
        :param verbose: Verbosity level to display tqdm progress bar
        :return: Training loss
        """
        self.model.train()
        train_loss = 0.
        i = 0
        epoch_loop = tqdm(self.train_loader, disable=verbose < VerboseLevel.TQDM, desc=f'* Epoch {epoch} - Training batches')
        for batch in epoch_loop:
            i += 1
            self.optimizer.zero_grad()

            batch_size_iter = batch.shape[0]
            batch = batch.to(self.device)
            batch_image = self.normalize_batch_images(batch, verbose=VerboseLevel.NONE) # too much verbosity else

            batch_t = torch.randint(0, timesteps, (batch_size_iter,), device=self.device).long()
            noise = torch.randn_like(batch_image)

            x_noisy = Diffusion.q_sample(constants_dict, batch_image, batch_t, noise=noise)
            predicted_noise = self.model(x_noisy, batch_t)

            loss = self.criterion(noise, predicted_noise)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            epoch_loop.set_postfix(loss=train_loss / i)
        train_loss /= len(self.train_loader)
        return train_loss

    def val_epoch(self, epoch: int, constants_dict: dict[str, Union[Tensor, float]], timesteps_val: Tensor,
                  verbose: VerboseLevel=VerboseLevel.TQDM) -> float:
        """
        Validate the model for one epoch
        :param epoch: Current epoch
        :param constants_dict: Constants for the model
        :param timesteps_val: Timesteps to use for the validation
        :param verbose: Verbosity level to display tqdm progress bar
        :return: Validation loss
        """
        self.model.eval()
        val_loss = 0.
        i = 0
        with torch.no_grad():
            epoch_loop = tqdm(self.val_loader, disable=verbose < VerboseLevel.TQDM, desc=f'* Epoch {epoch} - Validation batches')
            for batch in epoch_loop:
                i += 1
                batch_image = batch.to(self.device)
                batch_t = timesteps_val

                noise = torch.randn_like(batch_image)

                x_noisy = Diffusion.q_sample(constants_dict, batch_image, batch_t, noise=noise)
                predicted_noise = self.model(x_noisy, batch_t)

                loss = self.criterion(noise, predicted_noise)
                val_loss += loss.item()
                epoch_loop.set_postfix(loss=val_loss / i)
        val_loss /= len(self.val_loader)
        return val_loss

    @staticmethod
    def monitor_training(history: dict[int, dict[str, float]], best_epoch: int, save_losses_path: Optional[str],
                         logscale: bool, verbose: VerboseLevel=VerboseLevel.TQDM) -> None:
        """
        Monitor the training by plotting the losses
        :param history: Training history with train and validation losses
        :param best_epoch: Best epoch
        :param save_losses_path: Path to save the loss graph
        :param logscale: Whether to use a log scale for the losses
        :param verbose: Verbosity level to display tqdm progress bar
        :return: Training history with train and validation losses
        """
        train_losses = np.array([history[epoch]['train_loss'] for epoch in history])
        val_losses = np.array([history[epoch]['val_loss'] for epoch in history])
        epochs = np.array(history.keys())

        # After training, plot the train and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        if save_losses_path is not None:
            plt.savefig(save_losses_path.format('_loss-monitoring_'))
        if verbose >= VerboseLevel.DISPLAY:
            plt.show()
        else:
            plt.close()

        # Log-scale
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.plot(epochs, train_losses, label='Train Loss', color='blue')
        ax.plot(epochs, val_losses, label='Validation Loss', color='red')

        ax.scatter(x=best_epoch, y=val_losses[best_epoch], label='Best Epoch', color='k', marker='x', zorder=2)
        ax.vlines(x=best_epoch, ymin=-1, ymax=val_losses[best_epoch], colors='green', linestyles='dotted')
        ax.hlines(xmin=-10, xmax=best_epoch, y=val_losses[best_epoch], colors='green', linestyles='dotted')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if logscale:
            plt.yscale('log')
        x_ticks = np.append(ax.get_xticks(), best_epoch)
        y_ticks = np.append(ax.get_yticks(), val_losses[best_epoch])
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.xlim(-5, epochs[-1] + 5)
        if logscale:
            plt.ylim(10 ** np.floor(np.log10(min(val_losses.min(), train_losses.min()))),
                     10 ** np.ceil(np.log10(max(val_losses.max(), train_losses.max()))))
        else:
            plt.ylim(-0.1, max(val_losses.max(), train_losses.max()) + 0.1)
        plt.title('Training and Validation Loss Over Epochs (log-scale)')
        plt.legend()
        plt.grid(which='both', axis='both')
        if save_losses_path is not None:
            plt.savefig(save_losses_path.format('_loss-monitoring' + ('-log_' if logscale else '.jpg')))
        if verbose >= VerboseLevel.DISPLAY:
            plt.show()
        else:
            plt.close()