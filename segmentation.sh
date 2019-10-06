CONFIG_FILE=configs/resnet50_unet_stage2.yml
LOG_DIR=logs/resnet50_unet_stage2

catalyst-dl run \
--expdir src \
--config $CONFIG_FILE \
--logdir $LOG_DIR \
--resume logs/resnet50_unet/checkpoints/last_full.pth \
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
