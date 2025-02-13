import os
from typing import Tuple, List

import cv2
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.notebook import tqdm


class ACDCDataset(Dataset):
    def __init__(self, data_path, transform=None, mode='training', augment=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.augment = augment
        self.slices = self._get_slices()
        print(f'Loaded {len(self.slices)} slices in {self.mode} mode.')

    def _get_patients(self): 
        patients = []
        mode_path = os.path.join(self.data_path, self.mode)
        for patient in tqdm(os.listdir(mode_path), desc=f'Loading {self.mode} data'):
            patient_path = os.path.join(mode_path, patient)
            if os.path.isdir(patient_path):
                patients.append(patient_path)
        return patients

    def _parse_info(self, info_path):
        with open(info_path, 'r') as f:
            info = {}
            for line in f:
                key, value = line.strip().split(': ')
                info[key] = int(value) if key in ['ED', 'ES'] else value
        return info

    def _get_slices(self):
        slices = []
        patients = self._get_patients()
        for patient_path in tqdm(patients, desc='Loading slices'):
            #print(patient_path)
            info_path = os.path.join(patient_path, 'Info.cfg')
            info = self._parse_info(info_path)
            ed_frame = info['ED']
            es_frame = info['ES']

            for frame in [ed_frame, es_frame]:
                img_path = os.path.join(patient_path, f'{os.path.basename(patient_path)}_frame{frame:02d}.nii')
                gt_path = os.path.join(patient_path, f'{os.path.basename(patient_path)}_frame{frame:02d}_gt.nii')

                img = nib.load(img_path).get_fdata()
                gt = nib.load(gt_path).get_fdata()

                for slice_idx in range(img.shape[2]):
                    if self.augment:
                        augmented_slice = self.augment(img[:, :, slice_idx], gt[:, :, slice_idx], f'{os.path.basename(patient_path)}_frame{frame:02d}_slice{slice_idx:02d}')
                        for img_slice, gt_slice, id_slice in augmented_slice:
                            slices.append((img_slice, gt_slice, id_slice))
                    else:
                        slices.append((img[:, :, slice_idx], gt[:, :, slice_idx], f'{os.path.basename(patient_path)}_frame{frame:02d}_slice{slice_idx:02d}'))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img, gt, slice_id = self.slices[index]

        # Normalize the image to the range 0-255
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = torch.tensor(img, dtype=torch.uint8)

        gt = torch.tensor(gt, dtype=torch.uint8)

        if self.transform:
            img, gt = self.transform(img, gt)

        return (img, gt, slice_id)


def transform_image_and_mask(image: torch.Tensor, mask: torch.Tensor, image_size: int, padding: float=0.1,
                             make_dimension_mask: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:

    # Crop the image and mask to the interesting part of the base image
    def crop_to_content(img, mask):
        rows = torch.any(img, dim=1)
        cols = torch.any(img, dim=0)
        rmin, rmax = torch.where(rows)[0][[0, -1]]
        cmin, cmax = torch.where(cols)[0][[0, -1]]
        return img[rmin:rmax+1, cmin:cmax+1], mask[rmin:rmax+1, cmin:cmax+1]

    image, mask = crop_to_content(image, mask)

    # Determine the padding size
    padding_pixels = int(torch.floor(torch.tensor(padding * image_size // 2)))

    # Resize the image and mask, keeping the mask spatially aligned with the image
    def resize_with_padding(img, target_size, padding_pixels, mode):
        img = img.to(torch.float32)
        h, w = img.shape[:2]
        if h > w:
            new_h = target_size - 2 * padding_pixels
            new_w = int(new_h * w / h)
        else:
            new_w = target_size - 2 * padding_pixels
            new_h = int(new_w * h / w)

        resized_img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode=mode).squeeze(0).squeeze(0)
        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left

        padded_img = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return padded_img.to(torch.uint8)

    interpolation_image = 'bicubic' if image.shape[0] < image_size else 'area'
    interpolation_mask = 'nearest'

    image = resize_with_padding(image, image_size, padding_pixels, interpolation_image).unsqueeze(0).to(torch.float32) / 255.
    mask = resize_with_padding(mask, image_size, padding_pixels, interpolation_mask).to(torch.uint8)

    # Transform mask to 4 channels if needed
    if make_dimension_mask:
        mask = (1. * (torch.arange(4) == mask.unsqueeze(-1))).permute(2, 0, 1).to(torch.float32)

    return image, mask


def rotation_couple_images(image: torch.Tensor, mask: torch.Tensor, slice_id: str, max_angle: float, nb_rotations: int) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
    rotated_images = []
    for angle in np.linspace(-max_angle, max_angle, nb_rotations):
        if angle == 0:
            rotated_images.append((image, mask, slice_id + '_rot000'))
        else:
            rot_M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
            rotated_image = cv2.warpAffine(image, rot_M, (image.shape[1], image.shape[0]), borderValue=image.min())
            rotated_mask = cv2.warpAffine(mask, rot_M, (mask.shape[1], mask.shape[0]), borderValue=0.)
            rotated_images.append((rotated_image, rotated_mask, slice_id + f'_rot{angle:03.0f}'))
    return rotated_images