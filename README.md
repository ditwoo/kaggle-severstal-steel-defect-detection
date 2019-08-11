# [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

NOTE: Before runing experiments make sure that all required packages installed

## Runing experiment

```python
python3 train.py --config <path to config file (yaml or json)> --logs <folder to store logs>
```

`train.py` also have optional parameter `--device` (or `-d`) -- device id to use (ie GPU number).

When value passed with `--device` is less than 0 then CPU will be used for training.

