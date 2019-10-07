import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .utils import read_rgb_image, image2tensor


class DetectionDataset(Dataset):
    def __init__(
        self,
        images,
        target=None,
        folder: Path = None,
        transforms=None,
        target_long: bool = False,
    ):
        self.images = images
        self.labels = target
        self.folder: Path = folder
        self.transforms = transforms
        self.is_target_long = target_long

    def __len__(self):
        return len(self.images)

    def _get_image(self, idx):
        path2img = self.folder / self.images[idx] if self.folder else self.images[idx]
        img = read_rgb_image(path2img)
        return img

    def __getitem__(self, idx):
        img = self._get_image(idx)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        img = image2tensor(img)

        if self.labels is None:
            return img

        if self.is_target_long:
            lbl = torch.LongTensor(self.labels[idx])
        else:
            lbl = torch.FloatTensor([self.labels[idx]])
        return img, lbl


__all__ = ["DetectionDataset"]
