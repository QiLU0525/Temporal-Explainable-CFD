import argparse
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
from gen_admm_data import cal_inference
import random
import os
from scipy.sparse import coo_matrix
from scipy.io import mmread, mmwrite, mminfo
import json

def main():
    parser = argparse.ArgumentParser(description='NetDetect')
    parser.add_argument('--train_data', type=str, default='../train_test.csv')
    parser.add_argument('--fin_ratio', type=str, default='../findata_for_torch.csv')

    parser.add_argument('--metapath_dir', type=str, default="../meta-path/mat_by_year")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    '''读取训练测试数据集'''
    all_data = pd.read_csv(args.train_data,encoding='gb18030')

    num_pair =  all_data.shape[0] / 2 # 2865 对正负样本对
    random.seed(args.seed)
    random_indices = random.sample(range(0,num_pair),num_pair)

    train_ids = sorted([2*i for i in random_indices[:int(num_pair * 0.8)]] + [2*i+1 for i in random_indices[:int(num_pair * 0.8)]])
    valid_ids = sorted([2*i for i in random_indices[int(num_pair * 0.8):int(num_pair * 0.9)]] + [2*i+1 for i in random_indices[int(num_pair * 0.8):int(num_pair * 0.9)]])
    test_ids = sorted([2*i for i in random_indices[int(num_pair * 0.9):]] + [2*i+1 for i in random_indices[int(num_pair * 0.9):]])

    train_ids = torch.tensor(train_ids)
    valid_ids = torch.tensor(valid_ids)
    test_ids = torch.tensor(test_ids)


    '''读取financial ratio'''
    fin_ratios = pd.read_csv(args.fin_ratio,encoding='gb18030')
    fin_ratios = fin_ratios.set_index(['Stkcd','year'])
    fin_ratio_series = []
    for i in range(all_data.index):
        firm = all_data.loc[i,'Stkcd']
        time_series = json.loads(all_data.loc[i,'time_series'])
        # [-1,20]
        values = fin_ratios.loc[(firm,time_series),['AQI', 'AT', 'CFED', 'DSIR', 'DEPI', 'GMI', 'IG', 'LEV', 'OPM', 'RG', 'SG',
                     'SGEE', 'TextualSimilarity', 'PositiveVocabularyNum', 'NegativeVocabularyNum', 'TotalWordsNum',
                     'SentencesNum', 'WordsNum', 'EmotionTone1', 'EmotionTone2']].values().tolist()

        fin_ratio_series.append(values)
        # fin_ratio_series 的 length 应该和 all_data 行数相同
        print(len(fin_ratio_series))

    '''读取meta-path矩阵'''
    adj_dict = {}
    for year in range(2003, 2022):
        adj_dict[year] = {}
        path_dir = os.path.join(args.metapath_dir, str(year), 'metapath')
        for _, _, files in os.walk(path_dir):
            for f in files:
                adj_dict[year][f.replace('.mtx','')]= mmread(os.path.join(path_dir,f))

    '''保存数据到torch对象中'''
    op_dict = {}
    op_dict['Stkcd'] = all_data.Stkcd.tolist()
    op_dict['year'] = all_data.pred_year.tolist()
    op_dict['label'] = all_data.label.tolist()
    op_dict['seq'] = torch.tensor(json.loads(s) for s in all_data.time_series)

    op_dict['fin_ratio'] = torch.tensor(fin_ratio_series)
    op_dict['split_idx'] = {'train': train_ids, 'valid': valid_ids, 'test': test_ids}

    op_dict['adj'] = adj_dict

    # torch.save 保存的就是一个字典
    torch.save(op_dict, '{}.pt'.format())
