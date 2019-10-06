# [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

NOTE: Before runing experiments make sure that all required packages installed

## Runing experiment

```bash
python3 -m src.train --config <path to config file (yaml or json)> --logs <folder to store logs>
```

`train.py` also have optional parameter `--device` (or `-d`) -- device id to use (ie GPU number).

When value passed with `--device` is less than 0 then CPU will be used for training.

### Config structure

`train.py` expect configuration file with structure like this:


```
model: (* required)
    name: ...
    parameter1: ...
    parameter2: ...

optimizer: (* requred)
    name: ...
    parameter1: ...
    parameter2: ...

num epochs: ... (int, optional, default - 2)

random state: ... (int, optional, default - 2019)

num workers: ... (int, optional, default - 6)

loss: (* required)
    name: ...
    param1: ...
    param2: ...

minimize metric: ... (boolean, optional, default - True)

train: (* required)
    file: ...
    transforms: <albumentations dict of transformations>

validation: (* required)
    file: ...
    transforms: <albumentations dict of transformations>

test:
    folder: ...
```
