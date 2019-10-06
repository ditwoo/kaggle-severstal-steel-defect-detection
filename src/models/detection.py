import torch.nn as nn
from torchvision import models


_backbones = {
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201
}


class DenseNetDetector(nn.Module):
    def __init__(self, backbone: str = 'densenet121', num_classes: int = 1):
        super(DenseNetDetector, self).__init__()
        backbone_model = _backbones[backbone]
        self.model = backbone_model(pretrained=True)

        classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.model.classifier = classifier

    def forward(self, batch):
        return self.model(batch)
