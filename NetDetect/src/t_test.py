import torch
import os
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, wilcoxon
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

def T_test_by_y_pred(model_path_1, model_path_2, N_FOLDS):
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

def T_test(model_path_1, model_path_2, type='ind'):
    with open(f'results/{model_path_1}/result.json', 'r') as f:
        r1 = json.load(f)
    with open(f'results/{model_path_2}/result.json', 'r') as f:
        r2 = json.load(f)
    print(r1['results_list'])
    print(r2['results_list'])

    auc_1, auc_2 = sorted(r1['results_list']['auc']), sorted(r2['results_list']['auc'])
    fr_prec_1, fr_prec_2 = sorted(r1['results_list']['fr_prec']), sorted(r2['results_list']['fr_prec'])
    fr_re_1, fr_re_2 = sorted(r1['results_list']['fr_re']), sorted(r2['results_list']['fr_re'])
    le_prec_1, le_prec_2 = sorted(r1['results_list']['le_prec']), sorted(r2['results_list']['le_prec'])
    le_re_1, le_re_2 = sorted(r1['results_list']['le_re']), sorted(r2['results_list']['le_re'])
    mac_f1_1, mac_f1_2 = sorted(r1['results_list']['mac_f1']), sorted(r2['results_list']['mac_f1'])

    output = {}
    if type == 'rel':
        output['type'] = 'paired related'
        output['auc'] = [round(ttest_rel(auc_1, auc_2)[0], 3), ttest_rel(auc_1, auc_2)[1]]
        output['fr_prec'] = [round(ttest_rel(fr_prec_1, fr_prec_2)[0], 3), ttest_rel(fr_prec_1, fr_prec_2)[1]]
        output['fr_re'] = [round(ttest_rel(fr_re_1, fr_re_2)[0], 3), ttest_rel(fr_re_1, fr_re_2)[1]]
        output['le_prec'] = [round(ttest_rel(le_prec_1, le_prec_2)[0], 3), ttest_rel(le_prec_1, le_prec_2)[1]]
        output['le_re'] = [round(ttest_rel(le_re_1, le_re_2)[0], 3), ttest_rel(le_re_1, le_re_2)[1]]
        output['f1'] = [round(ttest_rel(mac_f1_1, mac_f1_2)[0], 3), ttest_rel(mac_f1_1, mac_f1_2)[1]]
    else: # type = 'ind'
        output['type'] = 'unpaired independent'
        output['auc'] = [round(ttest_rel(auc_1, auc_2)[0], 3), ttest_rel(auc_1, auc_2)[1]]
        output['fr_prec'] = [round(ttest_rel(fr_prec_1, fr_prec_2)[0], 3), ttest_rel(fr_prec_1, fr_prec_2)[1]]
        output['fr_re'] = [round(ttest_rel(fr_re_1, fr_re_2)[0], 3), ttest_rel(fr_re_1, fr_re_2)[1]]
        output['le_prec'] = [round(ttest_rel(le_prec_1, le_prec_2)[0], 3), ttest_rel(le_prec_1, le_prec_2)[1]]
        output['le_re'] = [round(ttest_rel(le_re_1, le_re_2)[0], 3), ttest_rel(le_re_1, le_re_2)[1]]
        output['f1'] = [round(ttest_rel(mac_f1_1, mac_f1_2)[0], 3), ttest_rel(mac_f1_1, mac_f1_2)[1]]

    return output

if __name__=='__main__':

    global corp_data, labels, SEED
    SEED = 2023
    corp_data = torch.load('ChinaCorp_1x_2023-06-07.pt')
    fraud_labels = corp_data['label']
    test_idx = corp_data['split_idx']['test']
    labels = fraud_labels[test_idx]
    # MODEL_1 = 'ml-SGB_1_2023-05-12'
    MODEL_1 = 'gat_5e-05_0.0_0.001_[2, 2]_256_30_[0]_64_64_None_2023-06-23'
    MODEL_2 = 'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[0]_512_64_None_2023-06-19'

    TEST_TYPE = 'rel' # or rel
    t_p = T_test(MODEL_1, MODEL_2, TEST_TYPE)
    config = {
        'model_1': MODEL_1,
        'model_2': MODEL_2,
        't_p': t_p,
        # 't_statistic': t_statistic,
        # 'p_value': p_value,
    }
    file_name = f"{MODEL_1.replace('ml-','').split('_')[0]}___{ MODEL_2.split('_')[0]}_{my_utils.get_date_postfix()}"
    with open('significance_test/{}.log'.format(file_name), 'w') as f:
        json.dump(config, f, indent=4)
