import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, roc_auc_score
import argparse
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from datetime import datetime
import random
import json
from time import time
import my_utils
import os
import collections

def main(save_path):
    X_train, X_test, Y_train, Y_test = load_data()
    my_utils.mkdir_p(save_path)
    class_weight = {0: 1 - args.pos_weight, 1: args.pos_weight}

    if args.model=='SVC':
        params = {
            'C':2.9,
            'degree': 3, # 没啥用
            'kernel':'rbf', # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
            'class_weight':args.pos_weight,
            'seed': args.seed,
        }
        model = SVC(C=params['C'], kernel=params['kernel'], degree= params['degree'], class_weight=class_weight, random_state=args.seed)
        # model = SVC(kernel=params['kernel'], class_weight=class_weight, random_state=args.seed)
        variant_name = f"{params['C']}_{params['degree']}_{params['kernel']}_{params['seed']}"

    elif args.model == 'LR':
        params = {
            'max_iter': 110,
            'class_weight': args.pos_weight,
            'seed': args.seed,
        }
        model = LogisticRegression(max_iter=params['max_iter'], class_weight=class_weight,
                    random_state=args.seed)
        variant_name = f"{params['max_iter']}_{params['class_weight']}_{params['seed']}"

        # model = LogisticRegression(class_weight=class_weight, random_state=args.seed, max_iter=200)

    elif args.model == 'NN':
        params = {
            'lr': 1e-3,
            'hidden_size': [16,16,16], #
            # 'solver': 'adam', # 'lbfgs', 'sgd'
            'alpha': 0.0001, # L2 penalty
            'batch_size': 200,
            'activation': 'relu', # 'identity', 'logistic', 'tanh', 'relu'
            'seed': args.seed
        }
        model = MLPClassifier(activation=params['activation'], learning_rate_init = params['lr'],
            hidden_layer_sizes = params['hidden_size'], alpha = params['alpha'],
            random_state = params['seed'], batch_size = params['batch_size']
        )
        variant_name = f"{params['lr']}_{params['hidden_size']}_{params['alpha']}_{params['batch_size']}_{params['activation']}_{params['seed']}"

    elif args.model == 'NB':
        params = {
            'priors': [0.028, 0.972],
            'var_smoothing': 5e-6,
        }
        model = GaussianNB(priors=params['priors'], var_smoothing=params['var_smoothing'])
        # model = GaussianNB(priors=[0.1, 0.9])
        variant_name = f"{params['priors']}_{params['var_smoothing']}"

    elif args.model == 'DT':
        # 'gini': CART,
        params = {
            'criterion': 'gini',
            'max_depth': 13, # > 13 结果都一样
            'min_samples_leaf': 10,  # 叶节点最少样本数
            'min_samples_split': 10,  # 节点划分的最小样本数, 没用
            'max_features': None, # 最大特征比例, 和 min_samples_leaf 对应
        }
        model = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'],
                                    min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'],
                                    max_features=params['max_features'])
        # model = DecisionTreeClassifier(criterion='gini', max_depth = 100, min_samples_leaf=1, ccp_alpha=0.0)
        variant_name = f"{params['criterion']}_{params['max_depth']}_{params['min_samples_leaf']}_{params['min_samples_split']}_{params['max_features']}"

    elif args.model == 'RF':
        params = {
            'criterion': 'gini',
            'max_depth': 10, # 改变max_depth变化不大
            'min_samples_leaf': 7, # 改变 min_samples_leaf 变化大
            'min_samples_split':10,
            'seed': args.seed, # 有用
            # 'max_leaf_nodes':,
            # 'max_features': 'auto', auto这个选项已经deprecated了
        }
        model = RandomForestClassifier(criterion=params['criterion'], max_depth=params['max_depth'],
                                       min_samples_leaf = params['min_samples_leaf'], min_samples_split=params['min_samples_split'],
                                       random_state= params['seed'])
        variant_name = f"{params['criterion']}_{params['max_depth']}_{params['min_samples_leaf']}_{params['min_samples_split']}_{params['seed']}"

    elif args.model == 'SGB':
        params = {
            'loss': 'exponential',
            'learning_rate': 1e-2,
            'max_depth': 9,  # 改变max_depth变化不大
            'min_samples_leaf': 12,  # 改变 min_samples_leaf 变化大
            'min_samples_split': 10,
            'seed': args.seed
        }
        model = GradientBoostingClassifier(loss=params['loss'], learning_rate=params['learning_rate'],
                                           max_depth=params['max_depth'],
                                           min_samples_leaf=params['min_samples_leaf'],
                                           min_samples_split=params['min_samples_split'],
                                           random_state=params['seed'],
                                       )
        variant_name = f"{params['loss']}_{params['learning_rate']}_{params['max_depth']}_{params['min_samples_leaf']}_{params['min_samples_split']}_{params['seed']}"

        # model = GradientBoostingClassifier(loss = 'exponential',learning_rate = 0.001, random_state=args.seed)

    model.fit(X_train, Y_train)
    start_t = time()
    Y_pred = model.predict(X_test)
    # np.save(os.path.join(save_path, 'y_pred.npy'), Y_pred)
    end_t = time()

    auc, acc, fr_prec, fr_recall, le_prec, le_recall, macro_f1, micro_f1, pos_neg_rate = my_utils.evaluate(Y_pred, Y_test)
    output_dict = {
        'infer_time': round(end_t - start_t, 4),
        'auc': auc, 'acc': acc, 'fr_prec': fr_prec, 'fr_re': fr_recall, 'le_prec': le_prec,
        'le_re': le_recall, 'mac_f1': macro_f1, 'pos_neg_rate': pos_neg_rate
    }
    print('test: {}'.format(output_dict))

    with open(os.path.join(save_path,f'{variant_name}.log'), 'w') as f:
        json.dump({'config': vars(args),
                   'params': params,
                   'result': output_dict,
        }, f, indent=4)

def calculate_avg_results(save_path):
    results = collections.defaultdict(list)
    for root, dirs, files in os.walk(save_path):
        log_files = [f for f in files if '.log' in f]
        for log in log_files:
            with open(os.path.join(save_path, log), 'r') as f:
                output = json.load(f)

            output = output['result']
            results['auc'].append(output['auc'])
            results['fr_prec'].append(output['fr_prec'])
            results['fr_re'].append(output['fr_re'])
            results['le_prec'].append(output['le_prec'])
            results['le_re'].append(output['le_re'])
            results['mac_f1'].append(output['mac_f1'])

    with open(os.path.join(save_path, 'result.json'), 'w') as f:
        json.dump({'variants': log_files,
                   'results_list': results,
                   'result_avg': {
                       'auc': round(np.mean(results['auc']), 4),
                       'fr_prec': round(np.mean(results['fr_prec']), 4),
                       'fr_re': round(np.mean(results['fr_re']), 4),
                       'le_prec': round(np.mean(results['le_prec']), 4),
                       'le_re': round(np.mean(results['le_re']), 4),
                       'mac_f1': round(np.mean(results['mac_f1']), 4),
                   }
        }, f, indent=4)
    return results

def load_data():
    corp_data = torch.load('ChinaCorp_{}x.pt'.format(args.MAX_POS_NEG_RATE))
    print(corp_data.keys())

    fin_seq = corp_data['fin_seq']
    nonfin_seq = corp_data['nonfin_seq']
    mda_seq = corp_data['mda_seq']

    fraud_labels = corp_data['label']
    seq_lens = corp_data['seq_len']
    train_idx, test_idx, valid_idx = corp_data['split_idx']['train'], corp_data['split_idx']['test'], \
                                     corp_data['split_idx']['valid']
    fin_embs = corp_data['fin_ratio']
    nfin_embs = corp_data['nonfin_ratio']
    #  mda_sentiments = corp_data['mda']
    mda_bert_embs = torch.load('mda_embs.pt').cpu()
    print('mda embedding shape: {}'.format(mda_bert_embs.shape))
    # 做非时序的，取出 t 时刻的下标
    # [5730,] 取最后一年的财务数据为特征
    t_fin_index = np.array(list(map(
        lambda i: fin_seq[i][seq_lens[i] - 1], range(0, len(seq_lens)))))
    t_nfin_index = np.array(list(map(
        lambda i: nonfin_seq[i][seq_lens[i] - 1], range(0, len(seq_lens)))))
    t_mda = np.array(list(map(
        lambda i: mda_seq[i][seq_lens[i] - 1], range(0, len(seq_lens)))))

    X = torch.cat(
        (fin_embs[t_fin_index], nfin_embs[t_nfin_index], mda_bert_embs[t_mda]), dim=-1
    ).data.numpy()
    X_train, X_test = X[train_idx], X[test_idx]

    # X_train = fin_emb[cur_year_fin_index[train_idx]].data.numpy()
    Y_train = fraud_labels[train_idx].data.numpy()
    # X_test = fin_emb[cur_year_fin_index[test_idx]].data.numpy()
    Y_test = fraud_labels[test_idx].data.numpy()
    return X_train, X_test, Y_train, Y_test



if __name__ == "__main__":
    parser = argparse.ArgumentParser("NetDetect")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--model", type=str, default="SGB", help="model")
    parser.add_argument("--MAX_POS_NEG_RATE", type=int, default=1, help="MAX_POS_NEG_RATE")
    parser.add_argument("--pos_weight", type=float, default=0.5, help="the weight of fraud instances")
    parser.add_argument("--use_class_wgt", type=bool, default=True, help="the weight of fraud instances")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 创建文件夹
    # save_path = 'results/ml-{}_{}_{}'.format(args.model, args.MAX_POS_NEG_RATE, datetime.now().date())
    # main(save_path)
    # save_path 也可以直接给出值
    save_path = 'results/ml-SVC_1_2023-05-11'
    calculate_avg_results(save_path)
