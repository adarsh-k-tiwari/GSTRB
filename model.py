# -*- coding: utf-8 -*-
"""GTSRB-Adarsh

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yorGOKNrEBH9rrCTmGcCu2UuLm55YqNr

## Personal notebook for GTSRB using pytorch

The German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. We cordially invite researchers from relevant fields to participate: The competition is designed to allow for participation without special domain knowledge. Our benchmark has the following properties:

1. Single-image, multi-class classification problem
2. More than 40 classes
3. More than 50,000 images in total
4. Large, lifelike database
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import GTSRB

class GTSRBModel(nn.Module):
    def __init__(self):
        super(GTSRBModel, self).__init__()

        # Building 3 deep convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size of flattened features
        self.flatten_size = 128 * 4 * 4
        # After 3 pooling layers: 28x28 -> 14x14 -> 7x7 -> 3x3

        # Building fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 43)

        # Adding a dropout layer
        self.dropout = nn.Dropout(0.5)

    # for forward propagation
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flatten_size)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        x = self.fc2(x)
        return x

model = GTSRBModel()