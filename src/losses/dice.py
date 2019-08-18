import torch
from torch import Tensor
from torch import nn
from torch.nn.modules.loss import _Loss

from typing import List


def soft_dice_score(pred: Tensor,
                    target: Tensor,
                    smooth=1e-3,
                    from_logits=False) -> Tensor:
    if from_logits:
        pred = pred.sigmoid()

#     import pdb; pdb.set_trace()
    
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) + smooth
    return 2 * intersection / union


class BinaryDiceLoss(_Loss):
    """
    Implementation of Dice loss for binary image segmentation task
    """

    def __init__(self, from_logits=True, weight=None, smooth=1e-3):
        super(BinaryDiceLoss, self).__init__()
        self.from_logits = from_logits
        self.weight = weight
        self.smooth = smooth

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        if y_true.sum() == 0:
            return 0

        dice = soft_dice_score(y_pred, y_true, from_logits=self.from_logits, smooth=self.smooth)
        loss = (1.0 - dice)

        return loss


class BinaryDiceLogLoss(_Loss):
    """Implementation of logarithic Dice loss for binary image segmentation task
    """

    def __init__(self, from_logits=True, weight=None, smooth=1e-3):
        super(BinaryDiceLogLoss, self).__init__()
        self.from_logits = from_logits
        self.weight = weight
        self.smooth = smooth

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        if y_true.sum() == 0:
            return 0

        iou = soft_dice_score(y_pred, y_true, from_logits=self.from_logits, smooth=self.smooth)
        loss = - torch.log(iou)
        return loss


class MulticlassDiceLoss(_Loss):
    """Implementation of Dice loss for multiclass (semantic) image segmentation task
    """

    def __init__(self, classes: List[int] = None, from_logits=True, weight=None, reduction='elementwise_mean'):
        super(MulticlassDiceLoss, self).__init__(reduction=reduction)
        self.classes = classes
        self.from_logits = from_logits
        self.weight = weight

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        if self.from_logits:
            y_pred = y_pred.softmax(dim=1)

        n_classes = y_pred.size(1)
        smooth = 1e-3

        loss = torch.zeros(n_classes, dtype=torch.float, device=y_pred.device)

        if self.classes is None:
            classes = range(n_classes)
        else:
            classes = self.classes

        if self.weight is None:
            weights = [1] * n_classes
        else:
            weights = self.weight

        for class_index, weight in zip(classes, weights):

            dice_target = (y_true == class_index).float()
            dice_output = y_pred[:, class_index, ...]

            num_preds = dice_target.long().sum()

            if num_preds == 0:
                loss[class_index] = 0
            else:
                dice = soft_dice_score(dice_output, dice_target, from_logits=False, smooth=smooth)
                loss[class_index] = (1.0 - dice) * weight

        if self.reduction == 'elementwise_mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()

        return loss


__all__ = ['BinaryDiceLoss', 'BinaryDiceLogLoss', 'MulticlassDiceLoss']
