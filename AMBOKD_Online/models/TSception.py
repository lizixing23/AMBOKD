import torch

from torch import nn
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout, Parameter,BatchNorm2d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class tsception(nn.Module):
    def __init__(self, num_features=30, num_classes=2):
        super(tsception, self).__init__()
        self.cnn_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=9,
                kernel_size=(1,125),
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8))
        )
        self.cnn_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=9,
                kernel_size=(1,62),
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8))
        )
        self.cnn_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=9,
                kernel_size=(1,31),
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8))
        )


        self.cnn_11 = nn.Sequential(
            nn.Conv2d(
                in_channels=9,
                out_channels=6,
                kernel_size=(59,1),
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        )
        self.cnn_22 = nn.Sequential(
            nn.Conv2d(
                in_channels=9,
                out_channels=6,
                kernel_size=(26,1),
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        )
        self.cnn_33 = nn.Sequential(
            nn.Conv2d(
                in_channels=9,
                out_channels=6,
                kernel_size=(26,1),
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.bn1 = BatchNorm2d(9)
        self.bn2 = BatchNorm2d(6)
        ##全连接
        self.fc_tsception = nn.Sequential(
            nn.Linear(756, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )#for TSception
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, len(data.y)

        x = x.view(batch, 59, -1)

        #SKNET1
        x = x.unsqueeze(dim=1)
        x1 = self.cnn_1(x)
        x2 = self.cnn_2(x)
        x3 = self.cnn_3(x)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.bn1(x)
        a22 = [1, 3, 5, 8, 10, 12, 14, 17, 19, 21, 23, 26, 28, 30, 32, 34, 36, 38, 40, 43, 45, 47, 50, 52, 54, 57]
        a33 = [2, 4, 6, 9, 11, 13, 15, 18, 20, 22, 24, 27, 29, 31, 33, 35, 37, 39, 41, 44, 46, 48, 51, 53, 55, 58]
        x11 = self.cnn_11(x)#200,6,1,42
        x_2 = x[:,:,a22,:]
        x_3 = x[:,:,a33,:]
        x22 = self.cnn_22(x_2)
        x33 = self.cnn_33(x_3)
        x = torch.cat([x11,x22,x33],dim=2)
        x = self.bn2(x)
        x = x.view(x.size()[0],-1)

        return x

class TSception(nn.Module):
    def __init__(self, num_classes=2):
        super(TSception, self).__init__()
        self.eeg_model_name = 'TSception'

        self.TSception = tsception()
        self.fc_tsception = nn.Sequential(
            nn.Linear(756, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )#for TSception
    def forward(self,data):

        if self.eeg_model_name == 'TSception':
            eeg_fea = self.TSception(data)#(batchsize,2)
            eeg_x = self.fc_tsception(eeg_fea)
        eeg_classes = eeg_x
        return eeg_classes