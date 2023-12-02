import datetime
import errno
import os
import pickle
import random
from pprint import pprint
import pandas as pd
import numpy as np
import torch
from scipy import io as sio
from scipy import sparse
import json
import dgl
from dgl.data.utils import _get_dgl_url, download, get_download_dir
from scipy.io import mmread, mmwrite, mminfo
import collections

KGE_path = '../KGE_pytorch/source/embed/entityEmbedding_{}.npy'
metapath_dir = '../meta-path/mat_by_year/{}/metapath/{}.mtx'
metapath_list = ['gu_qin_gu', 'gud_gud', 'gud_trans',
                 'mana_gud', 'mana_mana', 'mana_qin_gud',
                 'mana_qin_mana', 'sub_trans', 'trans_trans',
                 'trans_trans_trans']

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def setup_log_dir(args):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args["log_dir"], "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            args["model"], args["lr"], args["dropout"], args["weight_decay"], args["num_heads"], args["batch_size"], args["num_epochs"], args["seed"], args["MAX_SEQ"], date_postfix)
    )
    mkdir_p(log_dir)
    return log_dir

def get_date_postfix():
    """
    Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    # post_fix = "{}_{:02d}-{:02d}-{:02d}".format(dt.date(), dt.hour, dt.minute, dt.second)
    post_fix = dt.date()
    return post_fix

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Directory {} already exists.".format(path))
        else:
            raise

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.bool()

def graph_train_test_valid_split(n_listed_firm):
    random_indices = random.sample(range(0, n_listed_firm), n_listed_firm)

    g_train_idx = torch.from_numpy(np.sort( random_indices[:int(n_listed_firm * 0.8)])).long().squeeze(0)
    g_valid_idx = torch.from_numpy(
        np.sort(random_indices[int(n_listed_firm * 0.8):int(n_listed_firm * 0.9)])).long().squeeze(0)
    g_test_idx = torch.from_numpy(np.sort(random_indices[int(n_listed_firm * 0.9): ])).long().squeeze(0)

    train_mask = get_binary_mask(n_listed_firm, g_train_idx)
    valid_mask = get_binary_mask(n_listed_firm, g_valid_idx)
    test_mask = get_binary_mask(n_listed_firm, g_test_idx)

    return g_train_idx, g_valid_idx, g_test_idx, train_mask, valid_mask, test_mask

def generate_graph(year, listed_corp):
    '''
        读取TransD embedding, meta-path 邻接矩阵
        一个year一个图，每个图包含：
        各个meta-path图层的邻接矩阵、节点特征
        训练、测试、验证id
        训练、测试、验证mask
        所有节点的label
    '''

    raw_fraud = pd.read_csv('../frau_time_industry_money_til2022.csv', encoding='gb18030')
    n_listed_firm = len(listed_corp)  # 3124 -> 4111

    graph = dict()
    graph['num_class'] = 2
    '''
    为图节点准备三种初始向量：
    kg_embs, one_hot_embs, 以及随机生成的向量，但随机向量直接在模型里生成了，这里就不初始化了
    '''
    graph['kg_embs'] = torch.from_numpy(np.load(KGE_path.format(year))[:n_listed_firm]).float()
    # one_hot_embs 两种方法, 但是这个embedding太大了
    # torch.concat: dim=-1，直接把每一个meta-path的 [4111, 4111] 矩阵拼接成[4111, 4111 * 10]
    # torch.stack: 只是把 每一个meta-path的 [4111, 4111] 矩阵堆起来，变成 [10, 4111, 4111]
    # graph['one_hot_embs'] = torch.concat([torch.from_numpy(mmread(metapath_dir.format(year, p)).todense()) for p in metapath_list],dim=-1)
    # graph['one_hot_embs'] = torch.stack([torch.from_numpy(mmread(metapath_dir.format(year, p)).todense()) for p in metapath_list],dim=-1)

    graph['adj'] = [dgl.from_scipy(mmread(metapath_dir.format(year, p))) for p in metapath_list]
    # 二分类，两个类别
    graph['labels'] = torch.from_numpy(np.zeros(n_listed_firm, dtype=np.int64)).long()
    pos_label_index = raw_fraud[raw_fraud['fraud_year'] == year].Stkcd
    for i in pos_label_index:
        try:
            graph['labels'][listed_corp.index(i)] = 1
        except: # 违规的公司不在 listed_corp 列表里
            pass

    graph['g_train_idx'], graph['g_val_idx'], graph['g_test_idx'], \
            graph['train_mask'], graph['val_mask'], graph['test_mask'] = graph_train_test_valid_split(n_listed_firm)

    return graph


def split_train_valid_test(all_data, seed):
    '''
    :param all_data: train_test_x.csv 数据集
    :return: a dict, key 为 label 为 1 的样本的index，key 为该样本的index和对应的多个负样本的index
    不按时间划分训练、验证、预测集的话，就按照这个方法，生成的 pos2pos_neg 可以用来划分不平衡的数据集
    '''
    pos2pos_neg = collections.defaultdict(list)
    for i in all_data.index:
        label = all_data.loc[i,'label']
        if label == 1:
            pos_index = i
            pos2pos_neg[pos_index].append(pos_index)
        else: # label==0: negative sample
            pos2pos_neg[pos_index].append(i)

    pos_num = len(pos2pos_neg)
    random.seed(seed)
    random_indices = random.sample(pos2pos_neg.keys(), pos_num)
    train_ids = sorted([j for i in random_indices[:int(pos_num * 0.8)] for j in pos2pos_neg[i]])
    valid_ids = sorted([j for i in random_indices[int(pos_num * 0.8):int(pos_num * 0.9)] for j in pos2pos_neg[i]])
    test_ids = sorted([j for i in random_indices[int(pos_num * 0.9):] for j in pos2pos_neg[i]])

    return train_ids, valid_ids, test_ids

def load_corp_net():
    # 生成ChinaCorp.pt数据集
    MAX_POS_NEG_RATE = 5
    all_data = pd.read_csv('../train_test_x{}.csv'.format(MAX_POS_NEG_RATE), encoding='gb18030')
    with open('../meta-path/listed_firm.json', 'r') as f:
        listed_corp = json.load(f)

    # 如果不按照年份进行训练、验证、测试集划分的话，就调用 split_train_valid_test 函数
    # train_idx, valid_idx, test_idx = split_train_valid_test(all_data, seed=0)
    # 按照年份划分训练、验证、测试集
    train_idx = torch.tensor(sorted(all_data[all_data.pred_year < 2019].index)).long()
    valid_idx = torch.tensor(sorted(all_data[all_data.pred_year == 2019].index)).long()
    test_idx = torch.tensor(sorted(all_data[all_data.pred_year > 2019].index)).long()

    '''读取financial ratio'''
    fin_ratios = pd.read_excel('../会计信息质量-财务指标/findata_normalized.xlsx').set_index(['Symbol', 'EndDate'],inplace=False)
    nonfin_ratios = pd.read_csv('../非财务指标/nonfin_data_v2.csv', encoding='gb18030').set_index(['Symbol', 'year'], inplace=False)
    mda = pd.read_excel('../年报/管理层讨论与分析/mda_no_number.xlsx').set_index(['Scode', 'Year'], inplace=False)

    # 只保存index
    # 记录每一条数据的真实时间期数，因为后面要统一补到6期
    fin_series, non_fin_series, mda_series, node_emb_series, seq_lens = [], [], [], [], []

    print('开始对每一个样本的年份的财务指标、非财务指标和mda的情感特征建立索引')
    for i in all_data.index:
        firm = all_data.loc[i, 'Stkcd']
        # 年份从小到大
        time_series = sorted(json.loads(all_data.loc[i, 'time_series']))
        seq_lens.append(len(time_series))
        # 生成记录fin ratios, nonfin ratio, mda行下标的时间序列，总的financial ratios有 26876行
        fin_ids = np.array(list(map(
            lambda x: fin_ratios.index.tolist().index((firm, x)), time_series)))
        non_fin_ids = np.array(list(map(
            lambda x: nonfin_ratios.index.tolist().index((firm, x)), time_series)))
        mda_ids = np.array(list(map(
            lambda x: mda.index.tolist().index((firm, x)), time_series)))
        # 生成记录node embedding 行下标的时间序列，总的node embedding有 19年 * 4111 = 78109 行
        node_ids = np.array(list(map(
            lambda x: (x-2003) * len(listed_corp) + listed_corp.index(firm), time_series)))

        # 时间窗口补齐
        if len(time_series) < 6:
            fin_ids = np.concatenate(
                [fin_ids, np.full(shape=6 - len(time_series), fill_value=fin_ratios.shape[0])], axis=-1)
            non_fin_ids = np.concatenate(
                [non_fin_ids, np.full(shape=6 - len(time_series), fill_value=nonfin_ratios.shape[0])], axis=-1)
            mda_ids = np.concatenate(
                [mda_ids, np.full(shape=6 - len(time_series), fill_value=mda.shape[0])], axis=-1)
            node_ids = np.concatenate(
                [node_ids, np.full(shape=6 - len(time_series), fill_value= (2022-2003) * len(listed_corp))], axis=-1)

        fin_series.append(fin_ids.tolist())
        non_fin_series.append(non_fin_ids.tolist())
        mda_series.append(mda_ids.tolist())
        node_emb_series.append(node_ids.tolist())
        # fin_ratio_series 的 length 应该和 all_data 行数相同

    print('financial ratio series length: {}'.format(len(fin_series)))
    print('nonfin ratio series length: {}'.format(len(non_fin_series)))
    print('mda series length: {}'.format(len(mda_series)))
    print('node emb series length: {}'.format(len(node_emb_series)))

    graphs = {}
    for year in range(2003, 2022):
        graphs[year] = generate_graph(year, listed_corp)

    '''保存数据到torch对象中, 要保证数据格式是tensor'''
    op_dict = {}
    # op_dict['Stkcd'] = torch.from_numpy(all_data.Stkcd.values).long().squeeze(0)
    # op_dict['year'] = torch.from_numpy(all_data.pred_year.values).long().squeeze(0)
    op_dict['label'] = torch.from_numpy(all_data.label.values).long().squeeze(0)
    op_dict['split_idx'] = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # op_dict['seq'] = torch.tensor([json.loads(s) + [np.inf for i in range(6-len(json.loads(s)))] for s in all_data.time_series]).long() # 补齐时间窗口
    op_dict['fin_ratio'] = torch.tensor(fin_ratios.iloc[:,4:].values)
    op_dict['nonfin_ratio'] = torch.tensor(nonfin_ratios.values)

    # 这里只记录mda的情感特征，bert-mda的向量存在 mda_embs.pt 里面
    op_dict['mda'] = torch.tensor(mda.iloc[:,3:].values)

    op_dict['fin_seq'] = torch.tensor(fin_series)
    op_dict['nonfin_seq'] = torch.tensor(non_fin_series)
    op_dict['mda_seq'] = torch.tensor(mda_series)
    op_dict['node_seq'] = torch.tensor(node_emb_series)
    op_dict['seq_len'] = torch.tensor(seq_lens)

    op_dict['graphs'] = graphs
    torch.save(op_dict, 'ChinaCorp_{}x.pt'.format(MAX_POS_NEG_RATE))

class EarlyStopping(object):
    def __init__(self, dir_path, args):
        dt = datetime.datetime.now()
        # 保存模型记录，datetime.now() 可以获取当下的时间
        self.filepath = dir_path
        # "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(dt.date(), dt.hour, dt.minute, dt.second)
        self.args = args
        self.patience = args["patience"]
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.best_epoch = None
        self.early_stop = False

    def step(self, loss, acc, model, epoch, optimizer):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.best_epoch = epoch
            self.save_checkpoint(model, optimizer)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        elif acc >= self.best_acc:
            # 保存模型
            self.best_epoch = epoch
            self.save_checkpoint(model, optimizer)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model, optimizer):
        """Saves model when validation loss decreases."""
        # torch.save(model.state_dict(), self.filepath)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(self.filepath,'checkpoint.pth'))
        self.save_config()

    def load_checkpoint(self, model, optimizer, path=None):
        """Load the latest checkpoint."""
        if path==None:
            print('Loading checkpoint %s...' % self.filepath)
            checkpoint = torch.load(os.path.join(self.filepath,'checkpoint.pth'))
        else:
            print('Loading checkpoint %s...' % path)
            checkpoint = torch.load(path)
        # model.load_state_dict(torch.load(self.filepath))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save_config(self):
        with open(os.path.join(self.filepath,'config.json'), 'w') as f:
            json.dump(self.args, f, indent=4)


if __name__ == '__main__':
    load_corp_net()
