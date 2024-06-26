import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, roc_auc_score, f1_score, accuracy_score, recall_score
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def adjust_kd(ce_student,ce_teacher_a,ce_teacher_b):
    a = np.clip(ce_student/ce_teacher_a,0.1,10)
    b = np.clip(ce_student/ce_teacher_b,0.1,10)
    return a,b

def adjust_lr(sgd_opt,ratio_student,ratio_teacher_a,ratio_teacher_b,opt):

    lr_ratio = np.clip(((ratio_teacher_a+ratio_teacher_b)/(ratio_student*2))**opt.alpha,0.1,10)
    for params_group in sgd_opt.param_groups:
        params_group['lr'] = lr_ratio*opt.learning_rate

def train(epoch, train_dataset_loader, optimizer_eeg, optimizer_visual, optimizer_fusion, model_t_eeg, model_t_visual, model_s_fusion, loss_list, opt, **kwargs):
    # setting
    loss_func_CE, loss_func_KD, loss_func_CE_none, loss_func_KD_none = loss_list

    running_loss = 0.0
    prob_train_list = []
    true_train = []

    loss_fusion_ce_list = []
    loss_eeg_ce_list = []
    loss_pic_ce_list = []
    model_s_fusion.train()
    for i, batch in enumerate(train_dataset_loader):
        batch = batch.to(DEVICE)
        train_x_batch = batch
        train_y_batch = torch.tensor(batch.y).to(DEVICE)
        train_y_batch_softmax = np.argmax(batch.y, axis=1)
        true_train += train_y_batch.cpu().numpy().tolist()

        if opt.distill in ['None','E-KD','V-KD','MKD','EMKD','CA-MKD']:
            fusion_logit = model_s_fusion(train_x_batch).float()
            logsoftmax_fusion = F.log_softmax(fusion_logit / opt.T)
            softmax_fusion = F.softmax(fusion_logit / opt.T)
            loss_fusion = loss_func_CE(fusion_logit, torch.tensor(train_y_batch_softmax).to(DEVICE))

        if opt.model_s_fusion in ['CMM','MLB','AMM'] and opt.distill == 'None': # for fusion method CMM/MLB/AMM
            fusion_logit = model_s_fusion(train_x_batch).float()
            loss_fusion_total = loss_func_CE(fusion_logit, torch.tensor(train_y_batch_softmax).to(DEVICE))

        if opt.distill in ['E-KD','MKD','EMKD','CA-MKD']:
            eeg_logit = model_t_eeg(train_x_batch).float().detach()
            softmax_eeg = F.softmax(eeg_logit / opt.T)

        if opt.distill in ['V-KD','MKD','EMKD','CA-MKD']:
            pic_logit = model_t_visual(train_x_batch).float().detach()
            softmax_pic = F.softmax(pic_logit / opt.T)


        if opt.distill == 'E-KD':
            loss_fusion_eeg_kl = loss_func_KD(logsoftmax_fusion, torch.tensor(softmax_eeg.detach()).to(DEVICE).float())
            loss_fusion_total = loss_fusion + opt.T * opt.T * loss_fusion_eeg_kl

        if opt.distill == 'V-KD':
            loss_fusion_pic_kl = loss_func_KD(logsoftmax_fusion, torch.tensor(softmax_pic.detach()).to(DEVICE).float())
            loss_fusion_total = loss_fusion + opt.T * opt.T * loss_fusion_pic_kl

        if opt.distill == 'MKD':
            loss_fusion_eeg_kl = loss_func_KD(logsoftmax_fusion, torch.tensor(softmax_eeg.detach()).to(DEVICE).float())
            loss_fusion_pic_kl = loss_func_KD(logsoftmax_fusion, torch.tensor(softmax_pic.detach()).to(DEVICE).float())
            loss_fusion_total = loss_fusion + opt.T * opt.T * loss_fusion_pic_kl + opt.T * opt.T * loss_fusion_eeg_kl

        if opt.distill == 'EMKD':
            loss_fusion_eeg_kl = loss_func_KD_none(logsoftmax_fusion, torch.tensor(softmax_eeg.detach()).to(DEVICE).float())
            loss_fusion_pic_kl = loss_func_KD_none(logsoftmax_fusion, torch.tensor(softmax_pic.detach()).to(DEVICE).float())
            KL_loss = torch.stack((loss_fusion_eeg_kl.mean(axis=1), loss_fusion_pic_kl.mean(axis=1)), axis=1)
            dist_p_eeg = Categorical(torch.tensor(softmax_eeg))
            Entropy_eeg = dist_p_eeg.entropy()
            dist_p_pic = Categorical(torch.tensor(softmax_pic))
            Entropy_pic = dist_p_pic.entropy()
            CE_loss = torch.stack((Entropy_eeg, Entropy_pic), axis=1)  # (64, 2)
            softmax_z = F.softmax(CE_loss, dim=1).detach()
            weight_z = (1 - softmax_z) / (2 - 1)
            KL_loss_sum = torch.sum(KL_loss * weight_z, axis=1)
            loss_fusion_total = loss_fusion + 2 * opt.T * opt.T * KL_loss_sum.mean()

        if opt.distill == 'CA-MKD':
            loss_fusion_eeg_kl = loss_func_KD_none(logsoftmax_fusion,
                                          torch.tensor(softmax_eeg).to(DEVICE).float())  # (64, 2)
            loss_fusion_pic_kl = loss_func_KD_none(logsoftmax_fusion,
                                                torch.tensor(softmax_pic).to(DEVICE).float())
            KL_loss = torch.stack((loss_fusion_eeg_kl.mean(axis=1), loss_fusion_pic_kl.mean(axis=1)), axis=1)  # (2)
            CE_loss_eeg = loss_func_CE_none(eeg_logit / opt.T, torch.tensor(train_y_batch_softmax).to(DEVICE))
            CE_loss_pic = loss_func_CE_none(pic_logit / opt.T,
                                                   torch.tensor(train_y_batch_softmax).to(DEVICE))
            CE_loss = torch.stack((CE_loss_eeg, CE_loss_pic), axis=1)  # (64, 2)
            softmax_z = F.softmax(CE_loss, dim=1).detach()
            weight_z = (1 - softmax_z) / (2 - 1)
            KL_loss_sum = torch.sum(KL_loss * weight_z, axis=1)
            loss_fusion_total = loss_fusion + 2 * opt.T * opt.T * KL_loss_sum.mean()

        if opt.distill in ['None','E-KD','V-KD','MKD','EMKD','CA-MKD']:
            optimizer_fusion.zero_grad()
            loss_fusion_total.backward()
            optimizer_fusion.step()
            running_loss += loss_fusion_total.item()
            prob_train = fusion_logit.data.cpu().numpy()
            prob_train_list.extend(prob_train)

        if opt.distill in ['DML','KDCL']:
            eeg_logit = model_s_fusion.get_eeg_fea(train_x_batch).float()
            pic_logit = model_s_fusion.get_pic_fea(train_x_batch).float()
            logsoftmax_eeg = F.log_softmax(eeg_logit / opt.T)
            softmax_eeg = F.softmax(eeg_logit / opt.T)
            softmax_pic = F.softmax(pic_logit / opt.T)
            logsoftmax_pic = F.log_softmax(pic_logit / opt.T)
            loss_eeg = loss_func_CE(eeg_logit, torch.tensor(train_y_batch_softmax).to(DEVICE))
            loss_pic = loss_func_CE(pic_logit, torch.tensor(train_y_batch_softmax).to(DEVICE))

            if opt.distill == 'DML':

                loss_eeg_pic_kl = loss_func_KD(logsoftmax_eeg, torch.tensor(softmax_pic.detach()).to(DEVICE).float())
                loss_pic_eeg_kl = loss_func_KD(logsoftmax_pic, torch.tensor(softmax_eeg.detach()).to(DEVICE).float())
                loss_eeg_total = loss_eeg + loss_eeg_pic_kl
                loss_pic_total = loss_pic + loss_pic_eeg_kl
                optimizer_eeg.zero_grad()
                loss_eeg_total.backward()
                optimizer_visual.zero_grad()
                loss_pic_total.backward()
                optimizer_eeg.step()
                optimizer_visual.step()
                running_loss += loss_eeg_total.item()+loss_pic_total.item()

                prob_train_eeg = eeg_logit.data.cpu().numpy()
                prob_train_pic = pic_logit.data.cpu().numpy()
                prob_train = (prob_train_eeg + prob_train_pic) / 2
                prob_train_list.extend(prob_train)

            if opt.distill == 'KDCL':
                train_batch_x = model_s_fusion.KDCL(pic_logit.detach(), eeg_logit.detach()).float()  # fusion硬标签
                logsoftmax_fusion = F.log_softmax(train_batch_x / opt.T)
                softmax_fusion = F.softmax(train_batch_x / opt.T)
                loss_fusion = loss_func_CE(train_batch_x, torch.tensor(train_y_batch_softmax).to(DEVICE))
                loss_eeg_fusion_kl = loss_func_KD(logsoftmax_eeg, torch.tensor(softmax_fusion.detach()).to(DEVICE).float())
                loss_pic_fusion_kl = loss_func_KD(logsoftmax_pic, torch.tensor(softmax_fusion.detach()).to(DEVICE).float())
                loss_eeg_total = loss_eeg + opt.T * opt.T * loss_eeg_fusion_kl
                loss_pic_total = loss_pic + opt.T * opt.T * loss_pic_fusion_kl

                optimizer_eeg.zero_grad()
                loss_eeg_total.backward()
                optimizer_visual.zero_grad()
                loss_pic_total.backward()
                optimizer_fusion.zero_grad()
                loss_fusion.backward()  # 1
                optimizer_eeg.step()
                optimizer_visual.step()
                optimizer_fusion.step()
                running_loss += loss_fusion.item()
                prob_train = train_batch_x.data.cpu().numpy()
                prob_train_list.extend(prob_train)

        if opt.distill == 'EML':
            eeg_log, eeg_fea = model_s_fusion.get_eeg_fea(train_x_batch)
            pic_log, pic_fea = model_s_fusion.get_pic_fea(train_x_batch)
            eeg_fea = eeg_fea.float()
            pic_fea = pic_fea.float()
            eeg_log = eeg_log.float()
            pic_log = pic_log.float()
            train_batch_x = model_s_fusion.get_fusion_fea(eeg_fea.detach(), pic_fea.detach()).float()  # fusion硬标签
            EP_fea = (eeg_log + pic_log) / 2
            logsoftmax_fusion = F.log_softmax(train_batch_x / opt.T)
            softmax_fusion = F.softmax(train_batch_x / opt.T)
            logsoftmax_eeg = F.log_softmax(eeg_log / opt.T)
            logsoftmax_pic = F.log_softmax(pic_log / opt.T)
            softmax_EP = F.softmax(EP_fea / opt.T)
            loss_fusion = loss_func_CE(train_batch_x, torch.tensor(train_y_batch_softmax).to(DEVICE))
            loss_eeg = loss_func_CE(eeg_log, torch.tensor(train_y_batch_softmax).to(DEVICE))
            loss_pic = loss_func_CE(pic_log, torch.tensor(train_y_batch_softmax).to(DEVICE))
            loss_eeg_fusion_kl = loss_func_KD(logsoftmax_eeg, torch.tensor(softmax_fusion.detach()).to(DEVICE).float())
            loss_pic_fusion_kl = loss_func_KD(logsoftmax_pic, torch.tensor(softmax_fusion.detach()).to(DEVICE).float())
            loss_EP_fusion_kl = loss_func_KD(logsoftmax_fusion, torch.tensor(softmax_EP.detach()).to(DEVICE).float())
            loss_eeg_total = loss_eeg + opt.T * opt.T * loss_eeg_fusion_kl
            loss_pic_total = loss_pic + opt.T * opt.T * loss_pic_fusion_kl
            loss_fusion_total = loss_fusion + opt.T * opt.T * loss_EP_fusion_kl

            optimizer_eeg.zero_grad()
            loss_eeg_total.backward()
            optimizer_visual.zero_grad()
            loss_pic_total.backward()
            optimizer_fusion.zero_grad()
            loss_fusion_total.backward()  # 1
            optimizer_eeg.step()
            optimizer_visual.step()
            optimizer_fusion.step()
            running_loss += loss_fusion.item()
            prob_train = train_batch_x.data.cpu().numpy()
            prob_train_list.extend(prob_train)

        if opt.distill == 'MMOKD':
            eeg_fea = model_s_fusion.get_eeg_fea(train_x_batch).float()
            pic_fea = model_s_fusion.get_pic_fea(train_x_batch).float()
            train_batch_x = model_s_fusion.get_fusion_fea(train_x_batch).float()

            logsoftmax_fusion = F.log_softmax(train_batch_x / opt.T)
            softmax_fusion = F.softmax(train_batch_x / opt.T)
            logsoftmax_eeg = F.log_softmax(eeg_fea / opt.T)
            softmax_eeg = F.softmax(eeg_fea / opt.T)
            logsoftmax_pic = F.log_softmax(pic_fea / opt.T)
            softmax_pic = F.softmax(pic_fea / opt.T)

            loss_fusion = loss_func_CE(train_batch_x, torch.tensor(train_y_batch_softmax).to(DEVICE))
            loss_eeg = loss_func_CE(eeg_fea, torch.tensor(train_y_batch_softmax).to(DEVICE))
            loss_pic = loss_func_CE(pic_fea, torch.tensor(train_y_batch_softmax).to(DEVICE))
            loss_fusion_eeg_kl = loss_func_KD(logsoftmax_fusion, torch.tensor(softmax_eeg.detach()).to(DEVICE).float())
            loss_fusion_pic_kl = loss_func_KD(logsoftmax_fusion, torch.tensor(softmax_pic.detach()).to(DEVICE).float())
            loss_eeg_fusion_kl = loss_func_KD(logsoftmax_eeg, torch.tensor(softmax_fusion.detach()).to(DEVICE).float())
            loss_eeg_pic_kl = loss_func_KD(logsoftmax_eeg, torch.tensor(softmax_pic.detach()).to(DEVICE).float())
            loss_pic_fusion_kl = loss_func_KD(logsoftmax_pic, torch.tensor(softmax_fusion.detach()).to(DEVICE).float())
            loss_pic_eeg_kl = loss_func_KD(logsoftmax_pic, torch.tensor(softmax_eeg.detach()).to(DEVICE).float())

            if opt.IS_adjust_kd:
                fusion_a, fusion_b = adjust_kd(loss_fusion.item(), loss_pic.item(), loss_eeg.item())
                eeg_a, eeg_b = adjust_kd(loss_eeg.item(), loss_pic.item(), loss_fusion.item())
                pic_a, pic_b = adjust_kd(loss_pic.item(), loss_eeg.item(), loss_fusion.item())

            else:
                fusion_a, fusion_b = 1, 1
                eeg_a, eeg_b = 1, 1
                pic_a, pic_b = 1, 1

            loss_fusion_total = loss_fusion + fusion_a * opt.T * opt.T * loss_fusion_pic_kl + fusion_b * opt.T * opt.T * loss_fusion_eeg_kl
            loss_eeg_total = loss_eeg + eeg_a * opt.T * opt.T * loss_eeg_pic_kl + eeg_b * opt.T * opt.T * loss_eeg_fusion_kl
            loss_pic_total = loss_pic + pic_a * opt.T * opt.T * loss_pic_eeg_kl + pic_b * opt.T * opt.T * loss_pic_fusion_kl
            loss_fusion_ce_list.append(loss_fusion.item())
            loss_eeg_ce_list.append(loss_eeg.item())
            loss_pic_ce_list.append(loss_pic.item())

            if opt.IS_adjust_lr:
                loss_fusion_ce_epoch_first = kwargs.get('loss_fusion_ce_epoch_first', None)
                loss_eeg_ce_epoch_first = kwargs.get('loss_eeg_ce_epoch_first', None)
                loss_pic_ce_epoch_first = kwargs.get('loss_pic_ce_epoch_first', None)
                if epoch != 0:
                    ratio_fusion_loss = np.clip((loss_fusion_ce_epoch_first - loss_fusion.item()) / loss_fusion_ce_epoch_first, 0.1, 1)
                    ratio_eeg_loss = np.clip((loss_eeg_ce_epoch_first - loss_eeg.item()) / loss_eeg_ce_epoch_first, 0.1, 1)
                    ratio_pic_loss = np.clip((loss_pic_ce_epoch_first - loss_pic.item()) / loss_pic_ce_epoch_first, 0.1, 1)
                    adjust_lr(optimizer_eeg, ratio_eeg_loss, ratio_pic_loss, ratio_fusion_loss, opt)
                    adjust_lr(optimizer_visual, ratio_pic_loss, ratio_fusion_loss, ratio_fusion_loss, opt)
                    adjust_lr(optimizer_fusion, ratio_fusion_loss, ratio_pic_loss, ratio_eeg_loss, opt)
            optimizer_eeg.zero_grad()
            loss_eeg_total.backward()
            optimizer_visual.zero_grad()
            loss_pic_total.backward()
            optimizer_fusion.zero_grad()
            loss_fusion_total.backward()
            optimizer_eeg.step()
            optimizer_visual.step()
            optimizer_fusion.step()
            running_loss += loss_fusion_total.item()
            prob_train = train_batch_x.data.cpu().numpy()
            prob_train_list.extend(prob_train)

    # metrics calculate
    true_train = np.array(true_train).reshape([-1, opt.num_labels])
    prob_train_list = np.array(prob_train_list).reshape([-1, opt.num_labels])
    train_true = np.argmax(true_train, axis=1)
    train_prob = np.argmax(prob_train_list, axis=1)
    train_auc = roc_auc_score(true_train, prob_train_list)
    train_acc = accuracy_score(train_true, train_prob)

    if opt.IS_adjust_lr and epoch == 0:
        loss_fusion_ce_epoch_first = np.mean(loss_fusion_ce_list)
        loss_eeg_ce_epoch_first = np.mean(loss_eeg_ce_list)
        loss_pic_ce_epoch_first = np.mean(loss_pic_ce_list)
        return train_auc, train_acc, running_loss, loss_fusion_ce_epoch_first, loss_eeg_ce_epoch_first, loss_pic_ce_epoch_first
    else:
        return train_auc, train_acc, running_loss

def val(val_dataset_loader, model_s_fusion, opt):
    model_s_fusion.eval()
    # setting
    prob_val = []
    true_val = []

    with torch.no_grad():
        for i, batch in enumerate(val_dataset_loader):  # enumerate(给数据相应的索引id)
            batch = batch.to(DEVICE)
            val_x_batch = batch
            val_y_batch = torch.tensor(batch.y).to(DEVICE)
            true_val += val_y_batch.cpu().numpy().tolist()

            if opt.distill in ['None','E-KD','V-KD','MKD','EMKD','CA-MKD']:
                prob_v = model_s_fusion(val_x_batch).float()
                prob_v = prob_v.data.cpu().numpy()
                prob_val.extend(prob_v)

            if opt.distill == 'DML':
                prob_v_eeg = model_s_fusion.get_eeg_fea(val_x_batch).float().data.cpu().numpy()
                prob_v_pic = model_s_fusion.get_pic_fea(val_x_batch).float().data.cpu().numpy()
                prob_v = (prob_v_eeg + prob_v_pic) / 2
                prob_val.extend(prob_v)

            if opt.distill == 'KDCL':
                prob_v_eeg = model_s_fusion.get_eeg_fea(val_x_batch).float()
                prob_v_pic = model_s_fusion.get_pic_fea(val_x_batch).float()
                prob_v = model_s_fusion.KDCL(prob_v_pic, prob_v_eeg).float()
                prob_v = prob_v.data.cpu().numpy()
                prob_val.extend(prob_v)
            
            if opt.distill == 'EML':
                prob_v_eeg, prob_v_eegfea = model_s_fusion.get_eeg_fea(val_x_batch)
                prob_v_pic, prob_v_picfea = model_s_fusion.get_pic_fea(val_x_batch)
                prob_v = model_s_fusion.get_fusion_fea(prob_v_eegfea, prob_v_picfea).float()
                prob_v = prob_v.data.cpu().numpy()
                prob_val.extend(prob_v)

            if opt.distill == 'MMOKD':
                prob_v = model_s_fusion.get_fusion_fea(val_x_batch).float()
                # prob_v_eeg = fusion_torch.get_eeg_fea(val_x_batch).float().data.cpu().numpy() # eeg model trained under MMOKD strategy
                # prob_v_pic = fusion_torch.get_pic_fea(val_x_batch).float().data.cpu().numpy() # visual model trained under MMOKD strategy
                prob_v = prob_v.data.cpu().numpy()
                prob_val.extend(prob_v)

    ########################## calculate val results ########################
    true_val = np.array(true_val).reshape([-1, opt.num_labels])
    prob_val = np.array(prob_val).reshape([-1, opt.num_labels])

    val_true = np.argmax(true_val, axis=1)
    v_prob = np.argmax(prob_val, axis=1)

    val_f1 = f1_score(val_true, v_prob, average='weighted')
    val_precision = precision_score(val_true, v_prob, average='weighted')
    val_recall = recall_score(val_true, v_prob, average='weighted')
    val_auc = roc_auc_score(true_val, prob_val)
    val_acc = accuracy_score(val_true, v_prob)

    return val_acc, val_auc, val_precision, val_recall, val_f1