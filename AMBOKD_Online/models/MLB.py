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

def windows(data, size, step):
    start = 0
    while ((start + size) <= data.shape[0]):
        yield int(start), int(start + size)
        start += step


def segment_signal_without_transition(data, window_size, step):
    segments = []
    for (start, end) in windows(data, window_size, step):
        if (len(data[start:end]) == window_size):
            segments = segments + [data[start:end]]
    return np.array(segments)


def segment_dataset(X, window_size, step):
    win_x = []
    for i in range(X.shape[0]):
        win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
    win_x = np.array(win_x)
    return win_x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):

        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

        self.attn_dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(0.5)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, eeg,pic):
        key = self.key_layer(eeg)
        query = self.query_layer(pic)
        value = self.value_layer(pic)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)
        attention_probs = self.attn_dropout(attention_probs)
        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)

        ##transformer
        hidden_states = self.dense(context)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + pic)

        return hidden_states

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



class gcn(nn.Module):
    def __init__(self, num_features=30, num_classes=2):
        super(gcn, self).__init__()
        self.cnn_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=20,
                kernel_size=(1,125),
                stride=1,
                padding='same'
            ),
            BatchNorm2d(20),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        )
        self.cnn_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=20,
                kernel_size=(1,59),
                stride=1,
                padding='same'
            ),
            BatchNorm2d(20),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        )
        self.cnn_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=20,
                kernel_size=(1,31),
                stride=1,
                padding='same'
            ),
            BatchNorm2d(20),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        )


        self.cnn_11 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(59,2),
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        )

        # GCN
        self.gcn1 = GCNConv(150,100)
        weight = torch.randn(216, requires_grad=True)
        self.weight = Parameter(weight)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.bn2 = BatchNorm(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = BatchNorm2d(20)

        self.cnn_12 = nn.Sequential(
            nn.Conv1d(
                in_channels=59,
                out_channels=10,
                kernel_size=2,
                stride=1
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.cnn_13 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(1,1),
                stride=1
            ),
            BatchNorm2d(10),
            nn.LeakyReLU()
        )
        self.cnn_14 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(1,1),
                stride=1
            ),
            #nn.LeakyReLU()
        )

        self.rnn = nn.LSTM(
                input_size=490,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=0.5,
            )

        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, 256,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context # context : [batch_size, n_hidden * num_directions(=2)]
    def forward(self, data):

        x, edge_index, batch_size = data.x, data.edge_index, len(data.dataset_idx)
        window_size = 100
        step = 50
        window = 5
        x = x.view(batch_size, 59, -1)
        #滑窗
        x = x.cpu().numpy()
        x = np.transpose(x, [0, 2, 1])
        x = segment_dataset(x, window_size, step)
        x = np.transpose(x, [0, 3, 1, 2])  # 200,62,5,100
        x = torch.from_numpy(x)
        x = x.to(DEVICE)

       ##slice
        # x = x.reshape(200,62,3,-1)
        x = x.permute(0,2,1,3)
        x = x.reshape(batch_size*window,59,-1)
        #SKNET1
        x = x.unsqueeze_(dim=1)#200*5,1,62,100
        x1 = self.cnn_1(x)#200,9,62,22
        x2 = self.cnn_2(x)#200,9,62,29
        x3 = self.cnn_3(x)#200,9,62,33
        x = x1
        x = torch.cat([x, x2], dim=-1)
        x = torch.cat([x, x3], dim=-1)

        #GCN

        x = self.cnn_13(x)
        x = x.sum(dim=1)
        #GCN
        x = x.reshape(batch_size,window,59,-1)
        x = x.permute(0,2,1,3)
        x = x.reshape(batch_size*59,window,-1)
        weight = torch.relu(self.weight)
        weight = weight.repeat(batch_size)
        x_1 = x[:,0,:]
        x_1 = self.gcn1(x_1,edge_index,weight)
        x_1 = self.bn2(x_1)
        x_2 = x[:, 1, :]
        x_2 = self.gcn1(x_2, edge_index, weight)
        x_2 = self.bn2(x_2)
        x_3 = x[:, 2, :]
        x_3 = self.gcn1(x_3, edge_index, weight)
        x_3 = self.bn2(x_3)
        x_4 = x[:, 3, :]
        x_4 = self.gcn1(x_4, edge_index, weight)
        x_4 = self.bn2(x_4)
        x_5 = x[:, 4, :]
        x_5 = self.gcn1(x_5, edge_index, weight)
        x_5 = self.bn2(x_5)
        x = torch.cat([x_1,x_2,x_3,x_4,x_5],dim=1)

        x = x.reshape(batch_size,59,window,-1)
        x= x.permute(0,2,1,3)


        x = x.reshape(batch_size*window,59,-1).unsqueeze_(dim=1)
        x = self.cnn_11(x)

        #rnn
        x= x.reshape(batch_size,window,-1)
        out,(hn,_)= self.rnn(x)
        x = hn[-1]
        # att
        att_out = self.attention_net(out,x)
        eeg_fea = att_out.view(att_out.size(0), -1)


        return eeg_fea


class MLB(nn.Module):
    def __init__(self, num_classes=2):
        super(MLB, self).__init__()

        self.MCGRAM = gcn()
        self.efficientNet = efficient()

        self.pic_fusion = nn.Linear(1280, 256)
        self.fc_pic = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 2)
        )

        self.fc_final_mlb = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.mlb_fc_final = nn.Sequential(
            nn.Linear(256, 2)
        )

        self.pic_fc_mlb=nn.Sequential(
            nn.Linear(1280,64),
            nn.Tanh(),
            nn.Dropout(0.5)
        )
        self.eeg_fc_mlb=nn.Sequential(
            nn.Linear(256,64),
            nn.Tanh(),
            nn.Dropout(0.5)
        )
        self.fc_mlb = nn.Linear(64,num_classes)

        self.fc_eeg = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, data):
        pic_data, batch_size = data.pic, len(data.dataset_idx)
        pic_data = pic_data.view(batch_size, 3, 224, 224)

        eeg_fea = self.MCGRAM(data)
        pic_fea = self.efficientNet(pic_data)
        fusion_fea_eeg = self.eeg_fc_mlb(eeg_fea)
        fusion_fea_pic = self.pic_fc_mlb(pic_fea)
        fusion_fea = torch.mul(fusion_fea_eeg,fusion_fea_pic)
        fusion_fea = self.fc_mlb(fusion_fea)

        fusion_classes = fusion_fea
        return fusion_classes


