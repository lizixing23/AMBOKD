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



class efficient(nn.Module):
    def __init__(self, num_classes=2):
        super(efficient, self).__init__()
        efficient_b0 = EfficientNet.from_pretrained(model_name='efficientnet-b0',
                                                    weights_path='./models/effnet_pretrained/efficientnet-b0-33842d33c.pth')
        self.efficientNet = efficient_b0
        self.efficient_features = self.efficientNet.extract_features  ##([1, 1280, 7, 7])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,pic_data):

        pic_fea = self.efficient_features(pic_data)
        pic_fea = self.avg_pool(pic_fea)
        pic_fea = pic_fea.flatten(start_dim=1)
        return pic_fea



class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNet_b0, self).__init__()

        self.efficientNet = efficient()

        self.fc_pic_pic = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)
        )#for efficientnet



    def forward(self,data):
        pic_data, batch_size = data.pic, len(data.dataset_idx)
        pic_data = pic_data.view(batch_size, 3, 224, 224)

        pic_fea = self.efficientNet(pic_data)
        pic_x = self.fc_pic_pic(pic_fea)
        pic_classes = pic_x
        return pic_classes

