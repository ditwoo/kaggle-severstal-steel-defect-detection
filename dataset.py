import numpy as np
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset


"""
Mask & RLE transfomations source - https://www.kaggle.com/xhlulu/severstal-simple-2-step-pipeline
"""


def mask2rle(img) -> np.ndarray:
    """
    Convert mask (2 dimensional np.ndarray) to array of rles.
    """
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs


def rle2mask(rle, imgshape) -> np.ndarray:
    """
    Convert rle of image to a mask (2 dimensional np.ndarray).
    """
    rows, cols = imgshape[0], imgshape[1]
    rlePairs = rle.reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rlePairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    return img
    

def build_mask(rles, input_shape) -> np.ndarray:
    """
    Convert list of rles of image to ndarray of masks with dimensions equal to HxW
    Where each pixel will be 0 - nothing, 1 - 4 instance classes
    """
    depth = len(rles)
    mask = np.zeros(input_shape, dtype=np.uint8)
    for rle_cls, rle in enumerate(rles):
        if len(rle) != 0:
            mask[rle2mask(rle, input_shape) > 0] = rle_cls + 1
    return mask


def build_rles(masks) -> list:
    """
    Build rle from given mask.
    """
    depth, width, height = masks.shape
    rles = [mask2rle(masks[i, :, :]) for i in range(depth)]
    return rles


def load_image(image_path) -> np.ndarray:
#     img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
#     img = np.expand_dims(img, 2)
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def cv2_image_to_tensor(image: np.ndarray) -> torch.FloatTensor:
    """
    stealed from https://github.com/albu/albumentations/blob/master/albumentations/pytorch/transforms.py
    """
    tensor = torch.from_numpy(np.moveaxis(image / (255. if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    return tensor


def tensor_from_image(image: np.ndarray) -> torch.Tensor:
    """
    Updated version of:
        https://github.com/BloodAxe/pytorch-toolbelt/blob/d9fe8b6d2176e9bd0a346fd3b76e427358e9b344/pytorch_toolbelt/utils/torch_utils.py#L78
    """
    if image.dtype == np.uint8:
        image = (image / 255.).astype(np.float32)
        
    # (h, w, c) -> (c, w, h)
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


class SteelDataset(Dataset):
    def __init__(self, images, rles=None, folder: Path = None, transforms=None):
        """
        images: List[str] - list of images (only names)
        rles: List[List[List[int]]] - list of rles list
        folder: str or Path - folder with images
        transforms: albumentation transforms
        """
        self.images = images
        self.folder = folder
        self.rles = rles
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path2img = self.folder / self.images[idx] if self.folder else self.images[idx]
        img = load_image(path2img)
        
        if self.rles is None:
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            img = tensor_from_image(img)
            return img
        
        mask = build_mask(self.rles[idx], img.shape[:2])
        if self.transforms is not None:
            res = self.transforms(image=img, mask=mask)
            img, mask = res['image'], res['mask']
        
        img = tensor_from_image(img)
        mask = torch.from_numpy(mask).long()
        
        return img, mask
    

class SteelChannelDataset(Dataset):
    def __init__(self, images, rles=None, folder: Path = Path('.'), transforms=None):
        """
        images: List[str] - list of images (only names)
        rles: List[List[int]] - list of rles
        folder: str or Path - folder with images
        transforms: albumentation transforms
        """
        self.images = [load_image(folder / img_name) for img_name in images]
        self.rles = rles
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        if self.rles is None:
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            return img
        
        mask = rle2mask(self.rles[idx], img.shape[:2])   
        if self.transforms is not None:
            # mask (channels, height, width) -> (height, width, channels)
            res = self.transforms(image=img, mask=mask)
            # mask (height, width, channels) -> (channels, height, width)
            img, mask = res['image'], res['mask']

        return img, mask
