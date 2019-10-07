CONFIG_FILE=configs/resnet34_unet_stage4.yml
LOG_DIR=logs/resnet34_unet_stage4

catalyst-dl run \
--expdir src \
--config $CONFIG_FILE \
--logdir $LOG_DIR \
--verbose &&

python3 freeze_model.py \
--config $CONFIG_FILE \
--state ${LOG_DIR}/checkpoints/best.pth \
--shapes 1 3 256 1600 \
--out best.pt &&

python3 freeze_model.py \
--config $CONFIG_FILE \
--state ${LOG_DIR}/checkpoints/last.pth \
--shapes 1 3 256 1600 \
--out last.pt
