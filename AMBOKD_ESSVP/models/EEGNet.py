import torch
import torch.nn.functional as Fn
from torch import nn
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout, Parameter,BatchNorm2d
from torch_geometric.nn import GCNConv, global_sort_pool, global_add_pool, BatchNorm, GATConv
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
import torchvision.models as models
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class eegnet(nn.Module):
    def __init__(self, classes_num=2):
        super(eegnet, self).__init__()
        self.drop_out = 0.5  # 随机失活一半神经元，防止过拟合

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 32),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(59, 1),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

    def forward(self, data):
        x, batch_size = data.x, len(data.y)
        x = x.view(batch_size, 1, 59, -1)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)

        return x


class EEGNet(nn.Module):
    def __init__(self, num_classes=2):
        super(EEGNet, self).__init__()
        self.eegnet = eegnet()
        self.fc_eegnet = nn.Linear(144, num_classes)  # FC for EEGNet

    def forward(self, data):
        eeg_fea = self.eegnet(data)
        eeg_x = self.fc_eegnet(eeg_fea)
        return eeg_x