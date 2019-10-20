import yaml
import torch
from torch.nn import Module
from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner

from .models import (
    ResUnet, 
    UNetResNet, 
    LinkNet34, 
    DenseNetDetector, 
    ResnetDetector, 
    resnet34, 
    SCseUnet,
    ResUnetScSeDecoded,
    QUnet,
    ModelFromCheckpoint,
    EfficientUnet,
)
from .experiment import Experiment
from .optimizers import PlainRAdam, AdamW
from .metrics import MeanDiceCallback, AllAccuracyCallback, F1Callback, FBetaCallback
from .losses import (
    JointLoss,
    CCE,
    BinaryDiceLoss,
    BinaryDiceLogLoss,
    MulticlassDiceLoss,
    TverskyLoss,
    DiceAndBCE,
    FocalLossMultiChannel,
    FocalAndBCE,
)

registry.Model(ResUnet)
registry.Model(UNetResNet)
registry.Model(LinkNet34)
registry.Model(DenseNetDetector)
registry.Model(ResnetDetector)
registry.Model(resnet34)
registry.Model(SCseUnet)
registry.Model(ResUnetScSeDecoded)
registry.Model(QUnet)
registry.Model(EfficientUnet)
registry.Model(ModelFromCheckpoint)

registry.Callback(MeanDiceCallback)
registry.Callback(AllAccuracyCallback)
registry.Callback(F1Callback)
registry.Callback(FBetaCallback)

registry.Optimizer(PlainRAdam)
registry.Optimizer(AdamW)

registry.Criterion(JointLoss)
registry.Criterion(CCE)
registry.Criterion(BinaryDiceLoss)
registry.Criterion(BinaryDiceLogLoss)
registry.Criterion(MulticlassDiceLoss)
registry.Criterion(TverskyLoss)
registry.Criterion(DiceAndBCE)
registry.Criterion(FocalLossMultiChannel)
registry.Criterion(FocalAndBCE)
