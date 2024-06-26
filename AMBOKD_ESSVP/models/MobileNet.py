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




class mobilenet(nn.Module):
    def __init__(self, num_classes):
        super(mobilenet, self).__init__()
        mobile_net = models.MobileNetV2()
        modules = list(mobile_net.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)

    def forward(self, pic_data):

        pic_fea_vgg = self.convnet(pic_data)
        pic_fea_vgg = nn.functional.adaptive_avg_pool2d(pic_fea_vgg, (1, 1))
        pic_fea_vgg = torch.flatten(pic_fea_vgg, 1)

        return pic_fea_vgg

class MobileNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNet, self).__init__()

        self.pic_model_name = 'MobileNet'

        self.mobilenet = mobilenet(num_classes=2)

        self.fc_mobilenet = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)
        )#for mobilenet

    def forward(self,data):
        pic_data, batch_size = data.pic, len(data.y)
        pic_data = pic_data.view(batch_size, 3, 224, 224)

        if self.pic_model_name == 'MobileNet':
            pic_fea = self.mobilenet(pic_data)
            pic_x = self.fc_mobilenet(pic_fea)  # for mobilenet

        pic_classes = pic_x
        return pic_classes

