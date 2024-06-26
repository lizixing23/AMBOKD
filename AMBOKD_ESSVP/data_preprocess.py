import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
import os
import xml.etree.ElementTree as ET
from PIL import Image
import scipy.io as sio

img_shape = [640,360]# Simple scene image size
true_img_shape = [1280,720]# Complex scene image size
screen_shape = [2560,1440] # Screen resolution

def separate_data(ID_list, test_ID, data_len_list):
    train_ID = [str(i) for i in ID_list if i != test_ID]
    train_list = []
    for ID in train_ID:
        train_idx = ID_list.index(ID)
        train_idx_ = data_len_list[train_idx]
        train_idx__ = data_len_list[train_idx + 1]
        train = list(range(train_idx_, train_idx__))
        train_list.append(train)
    train_list = sum(train_list, [])
    val_list = list(range(data_len_list[ID_list.index(test_ID)], data_len_list[ID_list.index(test_ID) + 1]))
    return train_list, val_list

def data_load(data_path, ID_list, target_type, exp_type):
    train_X = []
    train_y = []
    train_pic = []
    data_len_list = []
    temp_len = 0
    data_len_list.append(temp_len)
    for ID in ID_list:
        print('loading data from ' + ID)
        matpath = data_path + '/'+ ID + '/eegdata/' + exp_type + target_type
        matList = os.listdir(matpath)
        for mat in matList:
            temp_name = matpath + '/' + mat
            temp_data = sio.loadmat(temp_name)
            temp_pic = temp_data['pic']
            train_pic.append(temp_pic)
            temp_X = temp_data['data']
            temp_X = temp_X.astype('float32')
            train_X.append(temp_X)
            temp_y = temp_data['label'].ravel()
            temp_y = np.asarray(pd.get_dummies(temp_y), dtype=np.int8)
            train_y.append(temp_y)
            temp_len = temp_len + len(temp_y)
        data_len_list.append(temp_len)
    train_X = np.dstack(train_X)  # 62*300*-1
    train_y = np.vstack(train_y)
    train_pic = np.vstack(train_pic)
    return train_X, train_y, train_pic, data_len_list


def crop_win_data(data_path, ID_list, target_type, exp_type):
    # load data
    train_X, train_y, train_pic, data_len_list = data_load(data_path, ID_list, target_type, exp_type)  # [channel, time_length, trial]
    train_X = np.transpose(train_X, [2, 0, 1])  # [trial,time_length,channel]##GCN

    return train_X, train_y, train_pic, data_len_list

def get_boxes(xml_path, tar_name):
    tree = ET.parse(xml_path)  # ET.parse()内要为完整相对路径
    root = tree.getroot()  # 类型为element
    boxes_screen = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        if obj_name == tar_name:
            # bndbox = obj.find('bndbox')

            xmin = int(obj[4][0].text)  # xmin
            ymin = int(float(obj[4][1].text))  # ymin
            xmax = int(obj[4][2].text)  # xmax
            ymax = int(obj[4][3].text)  # ymax
            box_screen = [xmin, xmax, ymin, ymax]
            boxes_screen.append(box_screen)
    return boxes_screen


def label_name(target_type):
    if target_type == 'armored':
        label_type = 'armored_car'
    if target_type == 'air':
        label_type = 'air_early_warning'
    if target_type == 'tank':
        label_type = 'tank'
    if target_type == 'no':
        label_type = 'no_target'
    if target_type == 'no_true':
        label_type = 'no_target'
    return label_type

def read_pic(pic_path, target_type,loader_type):
    if loader_type == "train_dataset":
        picpath = pic_path + target_type + '/' + 'images'
    if loader_type == "val_dataset" :
        picpath = pic_path + target_type + '/' + 'noise_0.2'##noise_0.2

    piclist = os.listdir(picpath)
    pics = [os.path.join(picpath, img) for img in piclist]

    return pics

def read_xml(pic_path, target_type):
    xmlpath = pic_path + target_type + '/annotations/xml'
    xmllist = os.listdir(xmlpath)
    label_type = label_name(target_type)
    boxes = []
    for each_xml in xmllist:
        each_xml_path = xmlpath + '/' + each_xml
        each_boxes = get_boxes(each_xml_path, label_type)
        boxes.append(each_boxes)
    return boxes
def box_toscreen(each_box,box_type):
    [xmin, xmax, ymin, ymax] = each_box
    if box_type == 2:
        pic_shape = img_shape
    elif box_type == 1:
        pic_shape = true_img_shape
    xmin_screen = (xmin - pic_shape[0] / 2) * box_type + screen_shape[0] / 2
    xmax_screen = (xmax - pic_shape[0] / 2) * box_type + screen_shape[0] / 2
    ymin_screen = (ymin - pic_shape[1] / 2) * box_type + screen_shape[1] / 2
    ymax_screen = (ymax - pic_shape[1] / 2) * box_type + screen_shape[1] / 2
    x_w = xmax_screen - xmin_screen
    y_w = ymax_screen - ymin_screen
    x_m = (xmax_screen + xmin_screen) / 2
    y_m = (ymax_screen + ymin_screen) / 2
    ##set the box size to 150*150
    if x_w < 150:
        xmax_screen = x_m + 75
        xmin_screen = x_m - 75
    if y_w < 150:
        ymax_screen = y_m + 75
        ymin_screen = y_m - 75

    box_screen = [xmin_screen, xmax_screen, ymin_screen, ymax_screen]
    return box_screen



class get_essvp_Data(Dataset):

    def __init__(self, X, Y, P, indices, loader_type, target_type, transform=None):
        # CAUTION - epochs and labels are memory-mapped, used as if they are in RAM.
        self.epochs = X
        self.labels = Y
        self.pic = P
        self.indices = indices
        self.loader_type = loader_type
        edgg = [[],[]]
        ##edge1
        edge = [[1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,
                 13,13,13,13,14,14,14,14,15,15,15,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,
                 22,22,22,22,23,23,23,23,24,24,24,25,25,25,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,
                 32,32,32,32,33,33,33,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,
                 42,42,42,43,43,43,44,44,44,44,44,45,45,45,45,45,46,46,46,46,47,47,47,47,48,48,48,49,49,49,50,50,50,50,51,51,51,51,
                 52,52,52,52,53,53,53,53,54,54,54,54,55,55,55,56,56,56,57,57,57,58,58,58,59,59,59],
                [2,3,1,4,6,1,5,7,2,6,9,11,3,7,10,12,2,4,13,15,3,5,14,16,9,10,17,4,11,8,18,5,8,12,19,4,13,9,20,5,10,14,21,6,11,15,22,7,12,16,23,
                 6,13,24,7,14,25,8,18,19,26,9,20,17,27,10,17,21,28,11,22,18,29,12,19,23,30,13,24,20,31,14,21,25,32,15,22,33,16,23,34,
                 17,27,28,18,26,29,35,19,26,30,36,20,31,27,37,21,28,32,38,22,33,29,39,23,30,34,40,24,31,41,25,32,42,27,37,44,43,28,38,45,43,29,39,35,44,30,36,40,45,
                 31,41,37,46,32,38,42,47,33,39,48,34,40,49,35,36,50,35,37,46,53,51,36,38,47,54,52,39,48,44,53,40,45,49,54,41,46,55,42,47,56,
                 43,51,52,57,44,53,50,58,45,50,54,59,44,46,51,55,45,47,52,56,48,53,58,49,54,59,50,58,59,51,55,57,52,56,57
                 ]
        ]

        edgg = (np.array(edge) - 1).tolist()
        edge_index = torch.tensor(edgg, dtype=torch.long)
        self.edge_index = edge_index
        ##edge1.

        ##*********************加载图片数据********************

        pic_path = './ESSVP_dataset/Image/'  # 图片读取路径
        if target_type in ['armored','air']:
            back_type = 'no'
        else:
            back_type = 'no_true'
        if target_type == 'tank':
            box_type = 1
        else:
            box_type = 2
        target_pic = read_pic(pic_path, target_type, loader_type)
        target_xml = read_xml(pic_path, target_type)
        back_pic = read_pic(pic_path, back_type, loader_type)
        back_xml = read_xml(pic_path, back_type)

        back_pic = back_pic
        back_xml = back_xml

        # *************图像数据***************
        pics_data = []
        for i in range(len(P)):
            pic_type, pic_idx, pic_loc = self.pic[i, 0], int(self.pic[i, 1]), self.pic[i, 2:6]

            if pic_type == 3 or pic_type == 4 or pic_type == 5: # armored car = 3, air = 4, tank = 5
                pic = target_pic[pic_idx]
                boxes = target_xml[pic_idx]
            if pic_type == 6 or pic_type == 7: # background with armored car & air = 6, background with tank = 7
                pic = back_pic[pic_idx]
                boxes = back_xml[pic_idx]


            box=[]
            for each_box in boxes:
                box_screen = np.array(box_toscreen(each_box, box_type))
                if (pic_loc == box_screen).all():
                    box = each_box

            if not box:
                print("The box is wrong.")
                print('pic_loc',pic_loc)
                print('boxes',boxes)
                print('pic_type',pic_type,'pic_idx',pic_idx)
                print('2222  ',i)
            [box_xmin, box_xmax, box_ymin, box_ymax] = box
            crop_box = [box_xmin, box_ymin, box_xmax, box_ymax]
            img_data = Image.open(pic)
            pic_data = img_data.crop(crop_box)
            pic_data = transform(pic_data)
            pic_data = pic_data.numpy()
            pics_data.append(pic_data)
        pics_data = np.array(pics_data)
        pics_data = torch.from_numpy(pics_data)

        self.pics_data = pics_data
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
#*************EEG_Graph****************
        node_features = self.epochs[idx]
        node_features = torch.from_numpy(node_features.reshape(59, 300))
#*************EEG_Graph****************
        data = Data(x=node_features,
                    edge_index=self.edge_index,
                    dataset_idx=idx,
                    y=self.labels[idx],
                    pic=self.pics_data[idx],
                    pp=self.pic[idx],
                    )
        return data

