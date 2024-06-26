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




class resnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet50, self).__init__()
        resnet_50 = models.resnet50(pretrained=False)
        modules = list(resnet_50.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)

    def forward(self, pic_data):

        pic_fea_res = self.convnet(pic_data)
        pic_fea_res = pic_fea_res.view(pic_fea_res.size(0), -1)

        return pic_fea_res

class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()

        self.pic_model_name = 'ResNet50'

        self.resnet50 = resnet50(num_classes=2)

        self.fc_resnet50 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )


    def forward(self,data):
        pic_data, batch_size = data.pic, len(data.y)
        pic_data = pic_data.view(batch_size, 3, 224, 224)
        # 脑电特征分类

        if self.pic_model_name == 'ResNet50':
            pic_fea = self.resnet50(pic_data)
            pic_x = self.fc_resnet50(pic_fea)  # for resnet50

        pic_classes = pic_x
        return pic_classes
