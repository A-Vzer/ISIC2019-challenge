import torch.nn as nn
import torch.nn.functional as F
import torch
from efficientnet_pytorch import utils


class EfficientEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=9):
        super(EfficientEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA._fc = nn.Identity()
        self.modelB._fc = nn.Identity()
        self.fc1 = utils.round_filters(1280, self.modelA._global_params)
        self.fc2 = utils.round_filters(1280, self.modelB._global_params)
        # Create new classifier
        self.classifier = nn.Linear(self.fc1 + self.fc2, nb_classes)

    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)

        x = self.classifier(F.relu(x))
        return x