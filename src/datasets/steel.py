import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .utils import read_rgb_image, image2tensor, build_mask, build_masks


class SteelDataset(Dataset):
    def __init__(self, images, target=None, folder: Path = None, transforms=None):
        """
        images: List[str] - list of images (only names)
        rles: List[List[List[int]]] - list of rles list
        folder: str or Path - folder with images
        transforms: albumentation transforms
        """
        self.images = images
        self.folder = folder
        self.rles = target
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)

    def _get_image(self, idx) -> np.ndarray:
        path2img = self.folder / self.images[idx] if self.folder else self.images[idx]
        img = read_rgb_image(path2img)
        return img
    
    def __getitem__(self, idx):
        img = self._get_image(idx)
        
        if self.rles is None:
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            img = image2tensor(img)
            return img
        
        mask = build_mask(self.rles[idx], img.shape[:2])
        if self.transforms is not None:
            res = self.transforms(image=img, mask=mask)
            img, mask = res['image'], res['mask']
        
        img = image2tensor(img)
        mask = torch.from_numpy(mask).long()
        
        return img, mask


class RAMSteelDataset(SteelDataset):
    def __init__(self, images, target=None, folder: Path = None, transforms=None):
        """
        images: List[str] - list of images (only names)
        rles: List[List[List[int]]] - list of rles list
        folder: str or Path - folder with images
        transforms: albumentation transforms
        """
        self.images = np.array([read_rgb_image(folder / img if folder else img)
                                for img in images], dtype=np.uint8)
        self.rles = target
        self.transforms = transforms

    def _get_image(self, idx) -> np.ndarray:
        return self.images[idx]


class ChannelviseSteelDataset(Dataset):
    def __init__(self, images, target=None, folder: Path = None, transforms=None):
        """
        images: List[str] - list of images (only names)
        rles: List[List[List[int]]] - list of rles list
        folder: str or Path - folder with images
        transforms: albumentation transforms
        """
        self.images = images
        self.folder = folder
        self.rles = target
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)

    def _get_image(self, idx) -> np.ndarray:
        path2img = self.folder / self.images[idx] if self.folder else self.images[idx]
        img = read_rgb_image(path2img)
        return img
    
    def __getitem__(self, idx):
        img = self._get_image(idx)
        
        if self.rles is None:
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            img = image2tensor(img)
            return img
        
        masks = build_masks(self.rles[idx], img.shape[:2])
        if self.transforms is not None:
            res = self.transforms(image=img, mask=masks)
            img, masks = res['image'], res['mask']
        
        img = image2tensor(img)
        masks = torch.from_numpy(masks).float()
        masks = masks.permute(2, 0, 1)
        
        return img, masks


__all__ = ['SteelDataset', 'RAMSteelDataset']
