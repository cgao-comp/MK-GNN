import torch
import argparse
import numpy as np

import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
import os
import torch.nn.functional as F
from collections import defaultdict

from models import TAMSGC
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'TAMSGC'
resume_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=True, help="using ddi")

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path

# 验证函数在验证数据集上进行验证
def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    # dill.dump(obj=smm_record, file=open('../data1/gamenet_records.pkl', 'wb'))
    dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))

    print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    if not os.path.exists(os.path.join("saved", model_name)):  # 保存模型训练参数
        os.makedirs(os.path.join("saved", model_name))

    # 加载数据
    data_path = '../data1/records_final.pkl'
    voc_path = '../data1/voc_final.pkl'

    ehr_adj_path = '../data1/ehr_adj_final.pkl'
    # ehr_adj_path = '../data/all_visit_adj_final.pkl'
    ddi_adj_path = '../data1/ddi_A_final.pkl'
    device = torch.device('cpu:0')  # 选择GPU训练还是CPU训练

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']


    # 划分数据集为训练集、验证集与测试集
    partition = int(len(data) * 2 / 3)
    data_train = data[:partition]
    eval_len = int(len(data[partition:]) / 2)
    data_test = data[partition:partition + eval_len]
    data_eval = data[partition+eval_len:]

    # 参数设置
    EPOCH = 40
    LR = 0.0001
    TEST = args.eval

    Neg_Loss = args.ddi
    DDI_IN_MEM = args.ddi
    TARGET_DDI = 0.05
    T = 0.5
    decay_weight = 0.85  # 0.85

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = TAMSGC(voc_size, ehr_adj, ddi_adj, emb_dim=64, device=device, ddi_in_memory=DDI_IN_MEM)  # 加载模型

    TEST = True # 选择是否测试
    if TEST:
        # 测试
        path = '../code/saved/TAMSGC/train'  # 测试模型参数保存的路径
        files = os.listdir(path)
        print(files)
        for file in files:
            print(file)
            model.load_state_dict(torch.load(open(path + '\\' + file, 'rb')))
            model.to(device=device)
            model.to()
            # print('parameters', get_n_params(model))
            # optimizer = Adam(list(model.parameters()), lr=LR)
            eval(model, data_test, voc_size, 0)
    else:
        # 训练
        model.to(device=device)
        model.to()
        print('parameters', get_n_params(model))
        optimizer = Adam(list(model.parameters()), lr=LR)  # 选择Adam优化器

        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()  # 训练开始时间
            model.train()  # 进入训练
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):  # 每位患者逐个进入训练
                for idx, adm in enumerate(input):  # 每位患者的每次就诊数据逐个进入训练
                    seq_input = input[:idx+1]
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1, batch_neg_loss = model(seq_input)
                    # loss计算损失
                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    # loss3 = F.multilabel_soft_margin_loss(F.sigmoid(target_output1),
                    #                                  torch.LongTensor(loss3_target).to(device))

                    if Neg_Loss:
                        target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                        target_output1[target_output1 >= 0.5] = 1
                        target_output1[target_output1 < 0.5] = 0
                        y_label = np.where(target_output1 == 1)[0]
                        current_ddi_rate = ddi_rate_score([[y_label]])
                        if current_ddi_rate <= TARGET_DDI:
                            loss = 0.9 * loss1 + 0.01 * loss3
                            prediction_loss_cnt += 1
                        else:
                            rnd = np.exp((TARGET_DDI - current_ddi_rate)/T)
                            if np.random.rand(1) < rnd:
                                loss = batch_neg_loss
                                neg_loss_cnt += 1
                            else:
                                loss = 0.9 * loss1 + 0.01 * loss3
                                prediction_loss_cnt += 1
                    else:
                        loss = 0.9 * loss1 + 0.01 * loss3

                    # loss = loss1
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)  # 反向传播
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))
            # annealing
            T *= decay_weight

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()  # 结束时间
            elapsed_time = (end_time - start_time) / 60  # 计算每一轮训练的时间
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))


            torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb')) # 保存模型的训练参数
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja


        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        # test
        # torch.save(model.state_dict(), open(
        #     os.path.join('saved', model_name, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()

