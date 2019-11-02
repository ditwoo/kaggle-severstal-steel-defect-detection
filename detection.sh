# CONFIG_FILE=configs/detection_folds/densenet121_fold_1.yml
# LOG_DIR=logs/detection_folds/densenet121_fold_1

# catalyst-dl run --expdir src --config $CONFIG_FILE --logdir $LOG_DIR --verbose

catalyst-dl run --expdir src --config configs/detection_folds/densenet121_fold_1.yml \
--logdir logs/detection_folds/densenet121_fold_1

echo "----------------------------------------------------------------------------------------------"
######################################

catalyst-dl run --expdir src --config configs/detection_folds/densenet121_fold_2.yml \
--logdir logs/detection_folds/densenet121_fold_2

echo "----------------------------------------------------------------------------------------------"
######################################

catalyst-dl run --expdir src --config configs/detection_folds/densenet121_fold_3.yml \
--logdir logs/detection_folds/densenet121_fold_3

echo "----------------------------------------------------------------------------------------------"
######################################

catalyst-dl run --expdir src --config configs/detection_folds/densenet121_fold_4.yml \
--logdir logs/detection_folds/densenet121_fold_4

echo "----------------------------------------------------------------------------------------------"
######################################

catalyst-dl run --expdir src --config configs/detection_folds/densenet121_fold_5.yml \
--logdir logs/detection_folds/densenet121_fold_5

# && python3 freeze_model.py --config $CONFIG_FILE \
# --state ${LOG_DIR}/checkpoints/best.pth \
# --shapes 1 3 256 256 --out best.pt

# && python3 freeze_model.py --config $CONFIG_FILE \
# --state ${LOG_DIR}/checkpoints/last.pth \
# --shapes 1 3 256 256 --out last.pt
