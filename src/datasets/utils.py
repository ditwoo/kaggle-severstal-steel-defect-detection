import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Union

"""
Mask & RLE transfomations source - https://www.kaggle.com/xhlulu/severstal-simple-2-step-pipeline
"""


def mask2rle(image: np.ndarray) -> np.ndarray:
    """
    Convert mask (2 dimensional np.ndarray) to array of rles.
    """
    pixels = image.T.flatten()
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
        img[index : index + length] = 255
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


def build_masks(rles, input_shape) -> np.ndarray:
    """
    Convert list of rles of image to ndarray of masks with dimensions equal to HxWxC
    Where each pixel will be 0 - nothing, 1 - 4 instance classes
    """
    depth = len(rles)
    masks = np.zeros((*input_shape, depth), dtype=np.uint8)

    for i, rle in enumerate(rles):
        if len(rle) != 0:
            masks[:, :, i] = rle2mask(rle, input_shape) > 0

    return masks


def build_rles(masks) -> list:
    """
    Build rle from given mask.
    """
    depth, width, height = masks.shape
    rles = [mask2rle(masks[i, :, :]) for i in range(depth)]
    return rles


def read_rgb_image(path: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_grayscale_image(path: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, 2)  # H*W -> C*H*W
    return img


def image2tensor(image: np.ndarray) -> torch.Tensor:
    """
    Updated version of:
        https://github.com/BloodAxe/pytorch-toolbelt/blob/d9fe8b6d2176e9bd0a346fd3b76e427358e9b344/pytorch_toolbelt/utils/torch_utils.py#L78
    """
    if image.dtype == np.uint8:
        image = (image / 255.0).astype(np.float32)

    # (h, w, c) -> (c, w, h)
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image
