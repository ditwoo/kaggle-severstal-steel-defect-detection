CONFIG_FILE=configs/detection.yml
LOG_DIR=logs/catalyst_detection

catalyst-dl run --expdir src --config $CONFIG_FILE --logdir $LOG_DIR --verbose --resume logs/catalyst/checkpoints/last.pth && 
python3 freeze_model.py --config $CONFIG_FILE --state ${LOG_DIR}/checkpoints/best.pth --shapes 1 3 256 256 --out best.pt &&
python3 freeze_model.py --config $CONFIG_FILE --state ${LOG_DIR}/checkpoints/last.pth --shapes 1 3 256 256 --out last.pt
