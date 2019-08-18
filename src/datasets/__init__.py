import numpy as np
from pandas import read_csv
from torch.utils.data import Dataset
from albumentations.core.serialization import from_dict

from .steel import SteelDataset, RAMSteelDataset
from .detection import DetectionDataset


datasets_map = {
    'SteelDataset': SteelDataset,
    'RAMSteelDataset': RAMSteelDataset,
    'DetectionDataset': DetectionDataset
}
classes = [f'cls{i}' for i in range(1, 5)]


def _rle_str2arr(rle_str: str) -> np.ndarray:
    res = []
    if not (rle_str != rle_str):
        res = list(map(int, rle_str.split(' ')))
    return np.array(res)


def load_transforms(transforms: dict):
    return from_dict(transforms)


def get_dataset(name: str, file: str, transforms: dict) -> SteelDataset:
    df = read_csv(file)
    dataset_cls = datasets_map[name]

    if name in {'SteelDataset', 'RAMSteelDataset'}:
        for _cls in classes:
            df[_cls] = df[_cls].apply(_rle_str2arr)
    
    data_transforms = load_transforms(transforms)
    
    dataset = dataset_cls(
        images=df['Image'].values,
        target=df[classes].values,
        transforms=data_transforms
    )
    return dataset

