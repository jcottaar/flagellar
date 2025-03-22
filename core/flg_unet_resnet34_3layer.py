import pandas as pd
import numpy as np
import scipy as sp
import dill # like pickle but more powerful
import itertools
import os
import copy
import json
import ndjson
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, fields
import enum
import typing
import zarr
import pathlib
import sklearn.neighbors
import cupy as cp
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import cupyx.signal
    import cupyx.scipy.ndimage
import functools
import hashlib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import time
import monai
import cc3d
try:
    import pytorch3dunet.unet3d.model
except:
    pass



'''
Helper functions and classes
'''

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=0.9)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet34_D_3D(nn.Module):
    def __init__(self, in_channels):
        super(ResNet34_D_3D, self).__init__()
        self.in_channels = in_channels

        # Modified initial layers: Replacing 7x7 conv + maxpool with three 3x3 convs
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, in_channels//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_channels//2, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(in_channels//2, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(in_channels, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(in_channels, 3)
        self.layer2 = self._make_layer(in_channels*2, 4, stride=2)
        self.layer3 = self._make_layer(in_channels*4, 6, stride=2)
        #self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels, momentum=0.9),
            )

        layers = []
        layers.append(BasicBlock3D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        #x4 = self.layer4(x3)

        return x1, x2, x3

class UNetResNet34_3D(nn.Module):
    def __init__(self, num_classes=2, in_channels=20):
        super(UNetResNet34_3D, self).__init__()

        self.encoder = ResNet34_D_3D(in_channels)

        # Decoder layers
        #self.decoder4 = self._decoder_block(512, 256)
        self.decoder3 = self._decoder_block(in_channels*4, in_channels*2)
        self.decoder2 = self._decoder_block(in_channels*2, in_channels)
        self.decoder1 = self._decoder_block(in_channels, in_channels)

        # Final output layer
        self.output_conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )

    def forward(self, x):
        # Encoder
        x1, x2, x3 = self.encoder(x)

        # Decoder
        #d4 = self.decoder4(x4) + x3
        d3 = self.decoder3(x3) + x2
        d2 = self.decoder2(d3) + x1
        d1 = self.decoder1(d2)

        # Output
        output = self.output_conv(d1)
        #output = torch.sigmoid(output)  # Normalize to [0, 1] for probability maps

        return output