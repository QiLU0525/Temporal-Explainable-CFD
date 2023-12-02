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

def evaluate(prediction, labels):
    result = classification_report(labels, prediction, target_names=['0', '1'], output_dict=True)
    auc = roc_auc_score(labels, prediction)
    accuracy = result['accuracy']
    # 欺诈样本的结果
    fraud_prec = result['1']['precision']
    fraud_recall = result['1']['recall']
    fraud_f1 = result['1']['f1-score']

    # 非欺诈样本的结果
    legit_prec = result['0']['precision']
    legit_recall = result['0']['recall']
    legit_f1 = result['0']['f1-score']

    # macro f1 计算出每一个类的Precison和Recall后计算F1，最后将F1平均，Macro-F1平等地看待各个类别，它的值会受到稀有类别的影响
    # macro_avg_prec 相当于 (fraud_prec + legit_prec)/2
    macro_avg_prec = result['macro avg']['precision']
    macro_avg_recall = result['macro avg']['recall']
    macro_avg_f1 = result['macro avg']['f1-score']

    # 每一个类根据其数量有不同的权重，相当于micro avg，是计算出所有类别总的Precision和Recall，然后计算F1，Micro-F1则更容易受到常见类别的影响。
    # weight_avg_prec 相当于 (fraud_prec * n_fraud + legit_prec * n_legit)/ (n_fraud+n_legit)
    weight_avg_prec = result['weighted avg']['precision']
    weight_avg_recall = result['weighted avg']['recall']
    weight_avg_f1 = result['weighted avg']['f1-score']

    pos_neg_rate = result['0']['support'] / result['1']['support']
    return round(auc, 4), round(accuracy, 4), round(fraud_prec, 4), round(fraud_recall, 4), round(legit_prec, 4), \
           round(legit_recall, 4), round(macro_avg_f1, 4), round(pos_neg_rate, 4)


def main():
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
            lambda i: fin_seq[i][seq_lens[i]-1], range(0,len(seq_lens)))))
    t_nfin_index = np.array(list(map(
            lambda i: nonfin_seq[i][seq_lens[i]-1], range(0,len(seq_lens)))))
    t_mda = np.array(list(map(
            lambda i: mda_seq[i][seq_lens[i]-1], range(0,len(seq_lens)))))

    X = torch.cat(
        (fin_embs[t_fin_index], nfin_embs[t_nfin_index], mda_bert_embs[t_mda]), dim=-1
    ).data.numpy()
    X_train, X_test = X[train_idx], X[test_idx]


    # X_train = fin_emb[cur_year_fin_index[train_idx]].data.numpy()
    Y_train = fraud_labels[train_idx].data.numpy()
    # X_test = fin_emb[cur_year_fin_index[test_idx]].data.numpy()
    Y_test = fraud_labels[test_idx].data.numpy()

    class_weight = {0: 1-args.pos_weight, 1: args.pos_weight}

    if args.model=='SVC':
        model = SVC(kernel='linear', class_weight=class_weight)
    elif args.model == 'LR':
        model = LogisticRegression(class_weight=class_weight, random_state=args.seed, max_iter=200)
    elif args.model == 'NN':
        args.use_class_wgt = False
        args.hidden_size = '[16,32,16]'
        model = MLPClassifier(activation='relu', hidden_layer_sizes=json.loads(args.hidden_size), alpha=0.0001)
    elif args.model == 'DT':
        # 'gini': CART,
        args.use_class_wgt = False # 表示没有使用 class_weight
        model = DecisionTreeClassifier(criterion='gini', max_depth = 100, min_samples_leaf=1, ccp_alpha=0.0)
    elif args.model == 'NB':
        args.use_class_wgt = False
        args.priors = [0.1,0.9]
        model = GaussianNB(priors=[0.1,0.9])
    elif args.model == 'RF':
        args.use_class_wgt = False
        model = RandomForestClassifier(criterion='gini', max_depth=100, random_state=args.seed,)
    elif args.model == 'SGB':
        args.use_class_wgt = False
        model = GradientBoostingClassifier(loss = 'exponential',learning_rate = 0.001, random_state=args.seed)
    else:
        model = SVC(kernel='rbf', class_weight=class_weight)
    model.fit(X_train, Y_train)
    start_t = time()
    Y_pred = model.predict(X_test)
    end_t = time()
    # accuracy = (Y_pred == Y_test).sum() / len(Y_pred)
    # f1 = f1_score(Y_test, Y_pred)
    # recall = recall_score(Y_test, Y_pred)
    # precision = precision_score(Y_test, Y_pred)
    # print(accuracy, precision, recall, f1)
    auc, acc, fr_prec, fr_recall, le_prec, le_recall, macro_f1, pos_neg_rate = evaluate(Y_pred, Y_test)
    output_dict = {
        'infer_time': round(end_t - start_t, 4),
        'auc': auc,
        'acc': acc,
        'fraud_prec': fr_prec,
        'fraud_recall': fr_recall,
        'legit_prec': le_prec,
        'legit_recall': le_recall,
        'macro_f1': macro_f1,
        'pos_neg_rate': pos_neg_rate
    }
    print('test: {}'.format(output_dict))
    with open('results/{}_{}_{}_{}_{}.log'.format(args.model, args.seed, args.MAX_POS_NEG_RATE, args.pos_weight, datetime.now().date()), 'w') as f:
        json.dump({'config':vars(args),
                   'result': output_dict,
        }, f, indent=4)

    # with open('results/{}_{}_{}_{}_{}.log'.format(args.model, args.seed, args.MAX_POS_NEG_RATE, args.pos_weight, datetime.now().date()), 'a+') as f:



if __name__ == "__main__":
    parser = argparse.ArgumentParser("NetDetect")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--model", type=str, default="NB", help="model")
    parser.add_argument("--MAX_POS_NEG_RATE", type=int, default=1, help="MAX_POS_NEG_RATE")
    parser.add_argument("--pos_weight", type=float, default=0.5, help="the weight of fraud instances")
    parser.add_argument("--use_class_wgt", type=bool, default=True, help="the weight of fraud instances")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    main()

