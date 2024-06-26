# -*- coding:utf-8 -*-
# ! /usr/bin/python3
import os
import numpy as np
import torch
import torch.nn as nn
import random
import time
from data_preprocess import get_essvp_Data, crop_win_data, separate_data
from models import teacher_model_dict, model_dict
from helper.loops import train, val
from torch_geometric.data import DataLoader
import warnings
import torchvision.transforms as transforms
import argparse

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_option():
    parser = argparse.ArgumentParser(description='Experimental parameter')
    parser.add_argument('--ID_list', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], help='List of IDs')
    parser.add_argument('--target_type', type=str, default='armored', choices=['armored', 'tank'], help='Target type') # use "tank" in the transfer experiment
    parser.add_argument('--exp_type', type=str, default='1_', choices=['1_', '2_'], help='Experiment type')
    parser.add_argument('--num_labels', type=int, default=15, help='Number of labels')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--training_epoch', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed_list', type=int, nargs='+', default=[1, 3, 6, 7, 9], help='List of seeds')
    parser.add_argument('--T', type=int, default=4, help='Temperature for KD distillation')
    parser.add_argument('--IS_adjust_lr', type=bool, default=True, help='Whether to adjust learning rate')
    parser.add_argument('--IS_adjust_kd', type=bool, default=True, help='Whether to adjust KD')
    parser.add_argument('--model_t_eeg', type=str, default='MCGRAM', choices=['MCGRAM', 'TSception', 'EEGNet'], help='EEG encoder & Teacher model')
    parser.add_argument('--model_t_visual', type=str, default='EfficientNet', choices=['EfficientNet', 'ResNet50', 'MobileNet'], help='Visual encoder & Teacher model')
    parser.add_argument('--model_s_fusion', type=str, default='MMOKD', choices=['CMM', 'MLB', 'AMM', 'DML', 'KDCL', 'EML', 'MMOKD'], help='Fusion module & Student model') # we use AMM with E-KD/V-KD/MKD/EMKD/CA-MKD in the experiment
    parser.add_argument('--distill', type=str, default='MMOKD', choices=['None', 'E-KD', 'V-KD', 'MKD', 'EMKD', 'CA-MKD', 'DML', 'KDCL', 'EML', 'MMOKD'], help='Knowledge distillation method') # CMM/MLB/AMM needs to use with 'None'
    parser.add_argument('--alpha', type=int, default=3, help='Hyper-parameter for DG block (AMB)')

    opt = parser.parse_args()

    # Set paths
    opt.data_path = './ESSVP_dataset/EEG'
    opt.model_path = './models'
    opt.save_path = './save'

    opt.model_name = f'T_{opt.model_t_eeg}_{opt.model_t_visual}_S_{opt.model_s_fusion}_distill_{opt.distill}_DK_{opt.IS_adjust_kd}_DG_{opt.IS_adjust_lr}'
    if opt.IS_adjust_lr:
        opt.model_name += f'_alpha_{opt.alpha}'
    opt.save_folder = f'{opt.save_path}/{opt.model_name}/cross/{opt.exp_type}/{opt.target_type}'
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def load_teacher_model(model_name, opt, k, test_ID):
    print('==> Loading teacher model', model_name)
    model_class, model_path = teacher_model_dict[model_name]
    model = model_class().to(DEVICE)
    model_param_dict = model.state_dict()
    model_param = torch.load(f'{model_path}/cross/{opt.exp_type}/{opt.target_type}/{k}/{test_ID}/gcn.ckpt')
    model_pretrained_dict = {k: v for k, v in model_param.items() if k in model_param_dict}
    model_param_dict.update(model_pretrained_dict)
    model.load_state_dict(model_param_dict)
    print('==> Done')
    return model


def main():
    opt = parse_option()
    ID_list = [str(id) for id in opt.ID_list]
    data_X, data_y, data_pic, data_len_list = crop_win_data(opt.data_path, ID_list, opt.target_type, opt.exp_type)

    loss_func_CE = nn.CrossEntropyLoss().to(DEVICE)
    loss_func_KD = nn.KLDivLoss().to(DEVICE)
    loss_func_CE_none = nn.CrossEntropyLoss(reduction='none').to(DEVICE)
    loss_func_KD_none = nn.KLDivLoss(reduction='none').to(DEVICE)

    loss_list = nn.ModuleList([loss_func_CE, loss_func_KD, loss_func_CE_none, loss_func_KD_none])

    input_height = 59  # EEG channels
    input_width = 300  # Sampling points

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for k in opt.seed_list:
        setup_seed(k)

        # Metrics
        bestmean_auc_list = []
        bestmean_acc_list = []
        bestmean_f1_list = []
        bestmean_recall_list = []
        bestmean_precision_list = []

        for test_ID in ID_list:  # 10-fold cross-validation
            best_auc = 0
            best_acc = 0
            best_f1 = 0
            best_recall = 0
            best_precision = 0
            loss_fusion_ce_epoch_first = 0
            loss_eeg_ce_epoch_first = 0
            loss_pic_ce_epoch_first = 0

            train_list, val_list = separate_data(ID_list, test_ID, data_len_list)
            train_x, train_y, train_pic = data_X[train_list], data_y[train_list], data_pic[train_list]
            val_x, val_y, val_pic = data_X[val_list], data_y[val_list], data_pic[val_list]

            # Load model
            model_t_eeg = model_t_visual = None
            if opt.distill in ['E-KD', 'MKD', 'EMKD', 'CA-MKD']:
                model_t_eeg = load_teacher_model(opt.model_t_eeg, opt, k, test_ID)
            if opt.distill in ['V-KD', 'MKD', 'EMKD', 'CA-MKD']:
                model_t_visual = load_teacher_model(opt.model_t_visual, opt, k, test_ID)

            model_class = model_dict[opt.model_s_fusion]
            model_s_fusion = model_class().to(DEVICE)

            # Initialize optimizers
            optimizer_eeg = optimizer_visual = optimizer_fusion = None
            if opt.distill in ['EML', 'DML', 'KDCL', 'MMOKD']:
                optimizer_eeg = torch.optim.Adam([
                    {'params': model_s_fusion.MCGRAM.parameters(), 'lr': opt.learning_rate},
                    {'params': model_s_fusion.fc_eeg.parameters(), 'lr': opt.learning_rate}
                ])
                optimizer_visual = torch.optim.Adam([
                    {'params': model_s_fusion.efficientNet.parameters(), 'lr': opt.learning_rate},
                    {'params': model_s_fusion.fc_pic_pic.parameters(), 'lr': opt.learning_rate}
                ])

                if opt.distill == 'EML':
                    optimizer_fusion = torch.optim.Adam([
                        {'params': model_s_fusion.deepconv.parameters(), 'lr': opt.learning_rate},
                        {'params': model_s_fusion.fc_final.parameters(), 'lr': opt.learning_rate}
                    ])
                elif opt.distill == 'KDCL':
                    optimizer_fusion = torch.optim.Adam([
                        {'params': model_s_fusion.KDCL_weight, 'lr': opt.learning_rate}
                    ])
                elif opt.distill in ['DML', 'MMOKD']:
                    optimizer_fusion = torch.optim.Adam([
                        {'params': model_s_fusion.pic_fc.parameters(), 'lr': opt.learning_rate},
                        {'params': model_s_fusion.eeg_fc.parameters(), 'lr': opt.learning_rate},
                        {'params': model_s_fusion.attention.parameters(), 'lr': opt.learning_rate},
                        {'params': model_s_fusion.fc_final.parameters(), 'lr': opt.learning_rate}
                    ])
            else:
                optimizer_fusion = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model_s_fusion.parameters()), lr=opt.learning_rate
                )

            # Prepare data
            train_x = train_x.reshape(-1, input_height * input_width)
            val_x = val_x.reshape(-1, input_height * input_width)

            train_dataset = get_essvp_Data(
                X=train_x, Y=train_y, P=train_pic, indices=train_list,
                loader_type="train_dataset", target_type=opt.target_type, transform=transform
            )
            val_dataset = get_essvp_Data(
                X=val_x, Y=val_y, P=val_pic, indices=val_list,
                loader_type="val_dataset", target_type=opt.target_type, transform=transform
            )

            train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

            for epoch in range(opt.training_epoch):
                epoch_start_time = time.time()
                print("\nEpoch", epoch + 1)
                if opt.IS_adjust_lr:
                    if epoch == 0:
                        train_auc, train_acc, running_loss, loss_fusion_ce_epoch_first, loss_eeg_ce_epoch_first, \
                        loss_pic_ce_epoch_first = train(epoch, train_loader, optimizer_eeg, optimizer_visual,
                                                        optimizer_fusion, model_t_eeg, model_t_visual, model_s_fusion, loss_list, opt)
                    else:
                        train_auc, train_acc, running_loss = train(
                            epoch, train_loader, optimizer_eeg, optimizer_visual, optimizer_fusion,
                            model_t_eeg, model_t_visual, model_s_fusion, loss_list, opt,
                            loss_fusion_ce_epoch_first=loss_fusion_ce_epoch_first,
                            loss_eeg_ce_epoch_first=loss_eeg_ce_epoch_first,
                            loss_pic_ce_epoch_first=loss_pic_ce_epoch_first
                        )
                else:
                    train_auc, train_acc, running_loss = train(
                        epoch, train_loader, optimizer_eeg, optimizer_visual, optimizer_fusion,
                        model_t_eeg, model_t_visual, model_s_fusion, loss_list, opt
                    )

                print(f'Train AUC: {train_auc:.4f}, Train Accuracy: {train_acc:.4f}, Loss: {running_loss:.4f}')

                val_acc, val_auc, val_precision, val_recall, val_f1 = val(val_loader, model_s_fusion, opt)

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_acc = val_acc
                    best_f1 = val_f1
                    best_recall = val_recall
                    best_precision = val_precision

                    model_save_path = os.path.join(opt.save_folder, f'{k}/{test_ID}')
                    os.makedirs(model_save_path, exist_ok=True)
                    torch.save(model_s_fusion.state_dict(), os.path.join(model_save_path, 'gcn.ckpt'))

                print(f"({time.asctime(time.localtime(time.time()))}) Epoch: {epoch + 1}\n"
                      f"Validation Accuracy: {val_acc:.4f}, Validation AUC: {val_auc:.4f}, Validation F1: {val_f1:.4f}, Best AUC: {best_auc:.4f}")
                epoch_end_time = time.time()
                print(f'Epoch took {epoch_end_time - epoch_start_time:.2f} seconds')

            bestmean_auc_list.append(best_auc)
            bestmean_acc_list.append(best_acc)
            bestmean_f1_list.append(best_f1)
            bestmean_recall_list.append(best_recall)
            bestmean_precision_list.append(best_precision)

        metric_save_path = os.path.join(opt.save_folder, str(k))
        np.savetxt(os.path.join(metric_save_path, 'best-auc.txt'), bestmean_auc_list)
        np.savetxt(os.path.join(metric_save_path, 'best-acc.txt'), bestmean_acc_list)
        np.savetxt(os.path.join(metric_save_path, 'best-f1.txt'), bestmean_f1_list)
        np.savetxt(os.path.join(metric_save_path, 'best-recall.txt'), bestmean_recall_list)
        np.savetxt(os.path.join(metric_save_path, 'best-precision.txt'), bestmean_precision_list)


if __name__ == '__main__':
    main()
