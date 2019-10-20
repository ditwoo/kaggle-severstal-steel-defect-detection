from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss

from .dice import MulticlassDiceLoss, BinaryDiceLogLoss, BinaryDiceLoss
from .pair import JointLoss, WeightedLoss
from .cce import CCE
from .tversky import TverskyLoss
from .focal import FocalLossMultiChannel


class DiceAndBCE(JointLoss):
    def __init__(self, dice_weight: float, bce_weight: float):
        super().__init__(
            MulticlassDiceLoss(), BCEWithLogitsLoss(), dice_weight, bce_weight
        )


class FocalAndBCE(JointLoss):
    def __init__(self, focal_weight: float, bce_weight: float):
        super().__init__(
            BCEWithLogitsLoss(), FocalLossMultiChannel(), bce_weight, focal_weight
        )
