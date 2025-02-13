import os
from functools import partial
from typing import Tuple

import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

from guided_diffusion.utils import VerboseLevel


class ACDCDataset(Dataset):
    def __init__(self, data_path, transform=None, mode='training', verbose: VerboseLevel=VerboseLevel.TQDM):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.verbose = verbose
        self.slices = self._get_slices()
        if verbose >= VerboseLevel.PRINT:
            print(f'Loaded {len(self.slices)} slices in {self.mode} mode.')

    def _get_patients(self):
        patients = []
        mode_path = os.path.join(self.data_path, self.mode)
        for patient in tqdm(os.listdir(mode_path), desc=f'Loading {self.mode} data', disable=self.verbose < VerboseLevel.TQDM):
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
        for patient_path in tqdm(patients, desc='Loading slices', disable=self.verbose < VerboseLevel.TQDM):
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


# Exemple usage
data_dir = '/path/to/data/acdc'
image_size = 128
batch_size = 4
make_dimension_mask = True # 4 channels boolean mask
verbose = VerboseLevel.PRINT

# Create dataset and data loader
transform_train = partial(transform_image_and_mask, image_size=image_size, padding=0.1, make_dimension_mask=True)
ds = ACDCDataset(data_path=data_dir, transform=transform_train, mode='training', verbose=verbose)
data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
data = iter(data_loader)

# Get next batch
images, masks, slice_ids = next(data)
print(f"images shape: {images.shape}")
print(f"masks shape: {masks.shape}")
print(f"slice_ids: {slice_ids}")