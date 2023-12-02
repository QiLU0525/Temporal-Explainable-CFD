import torch
import os
import numpy as np
from scipy.stats import ttest_rel, kstest, norm
import my_utils
import random
import json

def gen_groups(SEED, LEN, N_FOLDS):
    total = list(range(0, LEN))
    random.seed(SEED)
    random.shuffle(total)  # 打乱 total 中的数字顺序

    subsets = []  # 创建一个空列表，用于存储分组后的子列表

    per_size = int(LEN/N_FOLDS)
    for i in range(N_FOLDS):
        subset = total[i * per_size: min(LEN , (i + 1) * per_size)]  # 取出A中第i*10到(i+1)*10个数字，组成一个子列表
        subsets.append(subset)

    return subsets


def T_test(model_path_1, model_path_2, N_FOLDS):
    y_pred_1 = np.load(f'results/{model_path_1}/y_pred.npy')
    y_pred_2 = np.load(f'results/{model_path_2}/y_pred.npy')
    subsets = gen_groups(SEED, LEN=len(labels), N_FOLDS=N_FOLDS)

    # model 1:
    auc_1, fr_prec_1, fr_rec_1, le_prec_1, le_rec_1, f1_1 = [], [], [], [], [], []
    for subindex in subsets:
        auc, acc, fr_prec, fr_recall, le_prec, le_recall, macro_f1, micro_f1, pos_neg_rate = \
            my_utils.evaluate(y_pred_1[subindex], labels[subindex])
        auc_1.append(auc)
        fr_prec_1.append(fr_prec)
        fr_rec_1.append(fr_recall)
        le_prec_1.append(le_prec)
        le_rec_1.append(le_recall)
        f1_1.append(macro_f1)

    # model 2:
    auc_2, fr_prec_2, fr_rec_2, le_prec_2, le_rec_2, f1_2 = [], [], [], [], [], []
    for subindex in subsets:
        auc, acc, fr_prec, fr_recall, le_prec, le_recall, macro_f1, micro_f1, pos_neg_rate = \
            my_utils.evaluate(y_pred_2[subindex], labels[subindex])
        auc_2.append(auc)
        fr_prec_2.append(fr_prec)
        fr_rec_2.append(fr_recall)
        le_prec_2.append(le_prec)
        le_rec_2.append(le_recall)
        f1_2.append(macro_f1)

    t_statistic, p_value = {}, {}
    t_statistic['auc'], p_value['auc'] = ttest_rel(auc_1, auc_2)
    t_statistic['fr_prec'], p_value['fr_prec'] = ttest_rel(fr_prec_1, fr_prec_2)
    t_statistic['fr_recall'], p_value['fr_recall'] = ttest_rel(fr_rec_1, fr_rec_2)
    t_statistic['le_prec'], p_value['le_prec'] = ttest_rel(le_prec_1, le_prec_2)
    t_statistic['le_recall'], p_value['le_recall'] = ttest_rel(le_rec_1, le_rec_2)
    t_statistic['f1'], p_value['f1'] = ttest_rel(f1_1, f1_2)
    return t_statistic, p_value


if __name__=='__main__':
    global corp_data, labels, SEED
    SEED = 2023
    corp_data = torch.load('ChinaCorp_1x.pt')
    fraud_labels = corp_data['label']
    test_idx = corp_data['split_idx']['test']
    labels = fraud_labels[test_idx]
    MODEL_1 = 'ml-NB_2023_1_0.5_2023-05-10'
    MODEL_2 = 'base-han_5e-05_0.0_0.001_[2, 2]_256_30_2023_[0]_512_64_2023-05-10'

    t_statistic, p_value = T_test(MODEL_1, MODEL_2, N_FOLDS=10)
    config = {
        'model_1': MODEL_1,
        'model_2': MODEL_2,
        't_statistic': t_statistic,
        'p_value': p_value,
    }
    file_name = '{}___{}'.format( MODEL_1.replace('ml-','').split('_')[0], MODEL_2.split('_')[0] )
    with open('significance_test/{}.log'.format(file_name), 'w') as f:
        json.dump(config, f, indent=4)
