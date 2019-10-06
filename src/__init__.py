from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner

from .models import (
    ResUnet, 
    UNetResNet, 
    LinkNet34, 
    DenseNetDetector, 
    resnet34
)
from .experiment import Experiment
from .optimizers import (
    PlainRAdam, 
    AdamW
)
from .metrics import (
    AllAccuracyCallback,
    F1Callback,
    FBetaCallback,
)
from .losses import (
    JointLoss,
    CCE,
    BinaryDiceLoss,
    BinaryDiceLogLoss,
    MulticlassDiceLoss,
    TverskyLoss
)

registry.Model(ResUnet)
registry.Model(UNetResNet)
registry.Model(LinkNet34)
registry.Model(DenseNetDetector)
registry.Model(resnet34)

registry.Callback(AllAccuracyCallback)
registry.Callback(F1Callback)
registry.Callback(FBetaCallback)

registry.Optimizer(PlainRAdam)
registry.Optimizer(AdamW)