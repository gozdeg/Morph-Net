from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class custom_model(nn.Module):
    """
    Backbone wrapper that exposes a simple two-layer regression head.
    """

    def __init__(self, base_model_name, num_classes, feature_extract=False, use_pretrained=True):
        super().__init__()

        if base_model_name == "densenet":
            backbone = models.densenet121(pretrained=use_pretrained)
            num_ftrs = backbone.classifier.in_features
            self.base_model = backbone.features
        elif base_model_name == "resnet":
            backbone = models.resnet50(pretrained=use_pretrained)
            self.base_model = torch.nn.Sequential(*(list(backbone.children())[:-2]))
            num_ftrs = 2048
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")

        set_parameter_requires_grad(self.base_model, not feature_extract)

        self.hidden_fc = nn.Linear(num_ftrs, 256)
        self.relu = nn.ReLU(inplace=False)
        self.final_fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.hidden_fc(x)
        x = self.relu(x)
        x = self.final_fc(x)
        return x


def set_parameter_requires_grad(model, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad
