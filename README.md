# [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

NOTE: Before runing experiments make sure that all required packages installed <br>
NOTE: Link to [catalyst examples](https://github.com/catalyst-team/catalyst/tree/master/examples)

## About

Current pipeline looks:

```
detect images where all channels missed -> segment defects on images without missing channels
```

### Detection model

```bash
bash detection.sh
```

### Segmentation model

```bash
bash segmentation.sh
```