import torch
import argparse
import numpy as np
import os.path
import dill
import time
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from models import MKGNN
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params


def eval(model, data_eval, voc_size, epoch):
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
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}
        dill.dump(case_study, open(os.path.join('saved', model_name, str() + 'case_study.pkl'), 'wb'))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    llprint('\tJaccard: %.4f,  PRAUC: %.4f, AVG_F1: %.4f\n' % (
        np.mean(ja), np.mean(prauc), np.mean(avg_f1)
    ))

    return np.mean(ja), np.mean(prauc), np.mean(avg_f1)


def main():

    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    data_path = '../data2/prior.pkl'
    voc_path = '../data2/voc_final.pkl'
    ehr_adj_path = '../data2/ehr_adj_final.pkl'
    device = torch.device('cpu:0')
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]


    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1

    model = MKGNN(voc_size, ehr_adj, emb_dim=emd_dim, num_heads=2, device=device)

    TEST = True
    if TEST:
        #测试
        path = '../code/saved/MKGNN/train'
        files = os.listdir(path)
        test_history = defaultdict(list)
        for file in files:
            model.load_state_dict(torch.load(open(path + '\\' + file, 'rb')))
            model.to(device=device)
            ja, prauc, avg_f1 = eval(model, data_test, voc_size, 0)

            test_history['ja'].append(ja)
            test_history['avg_f1'].append(avg_f1)
            test_history['prauc'].append(prauc)
        dill.dump(test_history, open(os.path.join('saved', model_name, 'test_history.pkl'), 'wb'))

    else:
        model.to(device=device)
        print('parameters', get_n_params(model))
        optimizer = Adam(list(model.parameters()), lr=LR)

        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item
                    target_output1 = model(seq_input)
                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    # loss3 = F.multilabel_soft_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    lamda = 0.8
                    gama = 0.01

                    loss = lamda * loss1 + gama * loss3

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            ja, prauc, avg_f1 = eval(model, data_eval, voc_size, epoch)

            history['ja'].append(ja)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open(os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f.model' % (epoch, ja)), 'wb'))
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja

        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        print('best_epoch:', best_epoch)



if __name__ == '__main__':

    torch.manual_seed(1203)
    np.random.seed(1203)

    emd_dim = 128

    model_name = 'MKGNN'
    resume_name = ''
    print('***' * 20)
    print(model_name)

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
    parser.add_argument('--model_name', type=str, default=model_name, help="model name")
    parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')

    args = parser.parse_args()
    model_name = args.model_name
    resume_name = args.resume_path

    EPOCH = 15
    TEST = args.eval
    decay_weight = 0.85
    LR = 0.0002
    main()


