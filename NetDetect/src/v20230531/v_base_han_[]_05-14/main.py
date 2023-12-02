import torch
from torch.utils.data import TensorDataset, DataLoader
from my_utils import EarlyStopping, set_random_seed, setup_log_dir
import argparse
from model import Model
import numpy as np
import time
import os
import json
import scipy.sparse as sp
import dgl

def time_span_seq_align(corp_data, args):
    # 对应文章中6.2的需求，将时间窗口做多种切分，如只保留t期，只保留 t-5期，只保留 t-5，t-4期
    fin_seq, nfin_seq, mda_seq, node_seq, seq_lens = \
        corp_data['fin_seq'], corp_data['nonfin_seq'], corp_data['mda_seq'], corp_data['node_seq'], corp_data['seq_len']

    DEVICE = args["device"]
    # [-5, -4, ..., -1, 0] -> [-6, -5, ..., -2, -1]

    new_fin, new_nfin, new_mda, new_node, new_lens = [], [], [], [], []
    SUB_SEQ = torch.tensor(json.loads(args['MAX_SEQ'])) - 1
    if SUB_SEQ.tolist() == []:
        # [] 表示满长度
        new_fin, new_nfin, new_mda, new_node, new_lens = fin_seq, nfin_seq, mda_seq, node_seq, seq_lens
    elif -6 in SUB_SEQ:
        # 子序列从-5开始，early warning
        # SUB_SEQ: [-6, -5, -4, ...]
        for i in range(0, len(seq_lens)):
            new_fin.append(fin_seq[i][SUB_SEQ])
            new_nfin.append(nfin_seq[i][SUB_SEQ])
            new_mda.append(mda_seq[i][SUB_SEQ])
            new_node.append(node_seq[i][SUB_SEQ])
            new_lens.append(min(seq_lens[i], torch.tensor(SUB_SEQ.shape[0])))
        new_fin, new_nfin, new_mda, new_node, new_lens = torch.stack(new_fin), torch.stack(new_nfin), torch.stack(
            new_mda), torch.stack(new_node), torch.stack(new_lens)

    elif -1 in SUB_SEQ:
        # 子序列从0倒着回去，就近原则
        # SUB_SEQ: [..., -3, -2, -1]
        # SUB_SEQ = np.array([-3, -2, -1, 0]) - 1
        for i in range(0, len(seq_lens)):
            if seq_lens[i] >= SUB_SEQ.shape[0]:
                subseq = SUB_SEQ - (6 - seq_lens[i])
            else:  # seq_len < SUB_SEQ.shape[0]
                # 先往前回退 total_len - seq_len 个长度，再往前挪 SUB_SEQ.shape[0] - seq_len 个长度
                subseq = SUB_SEQ - (6 - seq_lens[i]) + (SUB_SEQ.shape[0] - seq_lens[i])
            new_fin.append(fin_seq[i][subseq])
            new_nfin.append(nfin_seq[i][subseq])
            new_mda.append(mda_seq[i][subseq])
            new_node.append(node_seq[i][subseq])
            new_lens.append(min(seq_lens[i], torch.tensor(SUB_SEQ.shape[0])))
        new_fin, new_nfin, new_mda, new_node, new_lens = torch.stack(new_fin), torch.stack(new_nfin), torch.stack(
            new_mda), torch.stack(new_node), torch.stack(new_lens)
    return new_fin.to(DEVICE), new_nfin.to(DEVICE), new_mda.to(DEVICE), new_node.to(DEVICE), new_lens.to(DEVICE)

def select_path(i, args, gs):
    # g: 股东，t: 关联交易关系，m: 高管，q：亲属关系，s：子公司关系
    path_list = ['gqg', 'gg', 'gt', 'mg', 'mm', 'mqg','mqm', 'st', 'tt', 'ttt']

    if args['path'] == 'total':
        adj = [adj.to(args["device"]) for adj in gs[2003 + i]['adj']]
    else:
        path_id = path_list.index(args['path'])
        adj = [gs[2003 + i]['adj'][path_id].to(args["device"])]
    return adj

def generate_model_name(args):
    SEQ_PART = '' # t, t-1, t-2, t-3, t-4, t-5
    EMB_PART = '' # kge, metapath,
    BASIC_PART = ''
    full_name = []
    # 先判断用没用传统特征
    if args['use_conven_feat'] is True:
        full_name.append('base')

    if args['use_metapath'] :
        if args['path'] == 'total':
            full_name.append('han')
        else:
            # 用单个 meta-path
            full_name.append('han({})'.format(args['path']))
    elif args['use_gat']:
        full_name.append('gat')

    if args['use_kge']:
        full_name.append('kge')
    return '-'.join(full_name)

def main(args):
    mda_bert_embs = torch.load('mda_embs.pt')
    corp_data = torch.load('ChinaCorp_{}x.pt'.format(args['MAX_POS_NEG_RATE']))
    # 里面的 mda 是统计词频的指标，没啥用，这里就不要了
    # print(corp_data.keys())

    # indices of financial, non-financial, mda indices
    fin_seq, nonfin_seq, mda_seq, node_seq, seq_lens = time_span_seq_align(corp_data, args)

    fraud_labels = corp_data['label'].to(args["device"])
    train_idx, test_idx, valid_idx = corp_data['split_idx']['train'], corp_data['split_idx']['test'], corp_data['split_idx']['valid']
    fin_emb = corp_data['fin_ratio'].to(args["device"])
    nonfin_emb = corp_data['nonfin_ratio'].to(args["device"])

    # mda_sentiment = corp_data['mda'].to(args["device"])
    gs = corp_data['graphs']
    # [19, 3124, 64] -> [19, 4111, 64]
    kg_embs = torch.stack([gs[y]['kg_embs'] for y in gs]).to(args["device"])

    n_listed_corps = kg_embs.shape[1]
    '''
    --------------------------------- 初始化模型 ---------------------------------
    '''
    model = Model(
        embs = [fin_emb, nonfin_emb, mda_bert_embs, kg_embs],
        # hidden_size=args["hidden_units"],
        out_size=2,
        num_path = 10 if args["path"]=='total' else 1,
        num_heads=args["num_heads"],
        dropout=args["dropout"],
        args=args,
    ).to(args["device"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"], eps=1e-4
    )
    logger = EarlyStopping(dir_path=args["log_dir"], args=args)

    if args["init_checkpoint"]:
        logger.load_checkpoint(model, optimizer, path=args["init_checkpoint"])

    for epoch in range(args["num_epochs"]):
        print('*' * 60 + 'Epoch {:d}'.format(epoch) + '*' * 60)
        with open(os.path.join(args["log_dir"],f'train_{args["seed"]}.log'), 'a+') as f:
            f.write("Epoch: {}\n".format(epoch))

        if args['use_gat']:
            model.train()
            for i in range(len(gs)): # 19 个自然年
                # 把10个meta path图合并，先转换成 scipy->numpy
                sum_adj = np.array([adj.adjacency_matrix(scipy_fmt="csr").toarray() for adj in gs[2003 + i]['adj']]).sum(0)
                # 再 numpy -> scipy -> dgl.graph
                sum_adj = dgl.from_scipy(sp_mat=sp.csr_matrix(sum_adj)).to(args["device"])
                node_indices = torch.tensor(range(i * n_listed_corps, (i + 1) * n_listed_corps)).to(args["device"])
                model.gat_infer(i, sum_adj, node_indices)

        elif args['use_metapath']:
            # han_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4, eps=1e-4)
            model.train()
            train_acc, train_losses, train_micro_f1, train_macro_f1 = np.zeros(len(gs)), np.zeros(len(gs)), np.zeros(len(gs)), np.zeros(len(gs))
            val_acc, val_losses, val_micro_f1, val_macro_f1 = np.zeros(len(gs)), np.zeros(len(gs)), np.zeros(len(gs)), np.zeros(len(gs))

            for i in range(len(gs)): # 19 个自然年
                adj = select_path(i, args, gs)
                # adj = [adj.to(args["device"]) for adj in gs[2003 + i]['adj']]

                labels = gs[2003 + i]['labels'].to(args["device"])
                # train_mask = gs[2003 + i]['train_mask'].to(args["device"])
                # val_mask = gs[2003 + i]['val_mask'].to(args["device"])
                # train HANs of different years, 尽量和HAN的原代码一样
                node_indices = torch.tensor(range(i * n_listed_corps, (i + 1) * n_listed_corps)).to(args["device"])
                HAN, logits,  = model.hans_infer(i, adj, node_indices)
                # logits: [4111, 2], labels: [4111], 2是类的个数

                '''loss = HAN.loss(logits[train_mask], labels[train_mask])
                optimizer.zero_grad() # 清空过往梯度
                loss.backward() # 反向传播，计算当前梯度
                optimizer.step() # 根据梯度更新网络参数
                _, train_micro_f1[i], train_macro_f1[i] = HAN.score(logits[train_mask], labels[train_mask])
                val_loss, _, val_micro_f1[i], val_macro_f1[i] = model.test_hans(i, adj, labels, val_mask, node_indices)
                train_losses[i], val_losses[i] = loss.item(), val_loss.item()
            
            han_output = {
                'train_loss': round(np.mean(train_losses), 4),
                'train_micro_f1': round(np.mean(train_micro_f1), 4),
                'train_macro_f1': round(np.mean(train_macro_f1), 4),
                'val_loss': round(np.mean(val_losses), 4),
                'val_micro_f1': round(np.mean(val_micro_f1), 4),
                'val_macro_f1': round(np.mean(val_macro_f1), 4),
            }
            print('------HAN------: {}'.format(han_output))
            with open(os.path.join(args["log_dir"],f'train_{args["seed"]}.log'), 'a+') as f:
                f.write("\t HAN: {}\n".format(han_output))'''

        # train NetDetect
        model.train()
        losses, logits = [], []
        train_loader = DataLoader(dataset=TensorDataset(
            node_seq[train_idx],
            fin_seq[train_idx],
            nonfin_seq[train_idx],
            mda_seq[train_idx],
            seq_lens[train_idx],
            fraud_labels[train_idx]),
            batch_size=args["batch_size"], shuffle=True)

        for i, (n_seq, f_seq, nf_seq, m_seq, seq_len, label) in enumerate(train_loader):
            logit = model(n_seq, f_seq, nf_seq, m_seq, seq_len)

            loss = model.loss(logit, label)
            # print(logit)
            # print(loss, loss.data, loss.cpu(), loss.cpu().data, loss.cpu().detach().numpy(), loss.cpu().item(), loss.item())
            losses.append(loss.item())
            logits.append(logit)
            loss.backward()
            time.sleep(10)
            optimizer.step()
            optimizer.zero_grad()

        # print(torch.cat(logits,dim=0).data.cpu().numpy())
        mean_losses = np.mean(losses)
        # print(torch.cat(logits,dim=0), fraud_labels[train_idx])
        train_results = model.score(torch.cat(logits,dim=0), fraud_labels[train_idx], mean_losses)

        print('------NeD------ Train : {}'.format(train_results))
        with open(os.path.join(args["log_dir"],f'train_{args["seed"]}.log'), 'a+') as f:
            f.write("\t NeD Train: {}\n".format(train_results))

        # validate epoch
        if epoch % args["valid_epoch"] == 0:
            val_results, _ = model.evaluate_(node_seq[valid_idx],
                                fin_seq[valid_idx],
                                nonfin_seq[valid_idx],
                                mda_seq[valid_idx],
                                seq_lens[valid_idx],
                                fraud_labels[valid_idx]
                )
            print('------NeD------ Valid : {}'.format(val_results) )
            with open(os.path.join(args["log_dir"], f'train_{args["seed"]}.log'), 'a+') as f:
                f.write("\t NeD Valid: {}\n".format(val_results))

            # test stage
            test_results, y_pred = model.evaluate_(
                    node_seq[test_idx],
                    fin_seq[test_idx],
                    nonfin_seq[test_idx],
                    mda_seq[test_idx],
                    seq_lens[test_idx],
                    fraud_labels[test_idx]
                )

            # 验证结束以后保存模型, early_stop 为true或者false
            # 用于保存模型，生成 config.json，保存模型用test acc，这样保存下来的模型就是最好的test效果
            early_stop = logger.step(loss=test_results['loss'], acc=test_results['acc'], model=model, epoch=epoch, optimizer=optimizer)
            # if logger.best_epoch == epoch:
                # 当前轮次更新为最佳轮次
                # 把预测的 label 存下来
                # np.save(os.path.join(args["log_dir"], 'y_pred.npy'), y_pred)
            print('------NeD------ Test : {}'.format(test_results))
            with open(os.path.join(args["log_dir"], f'train_{args["seed"]}.log'), 'a+') as f:
                f.write("\t NeD Test: {}\n".format(test_results))

    # test HANs of different years...

    # 将最佳轮的模型参数读入
    logger.load_checkpoint(model, optimizer)
    # best test
    # loss, auc, acc, fr_prec, fr_recall, le_prec, le_recall, macro_f1, micro_f1, pos_neg_rate = \

    best_test_results, _ = model.evaluate_(
            node_seq[test_idx],
            fin_seq[test_idx],
            nonfin_seq[test_idx],
            mda_seq[test_idx],
            seq_lens[test_idx],
            fraud_labels[test_idx]
        )

    print("BEST Epoch: {}".format(logger.best_epoch))
    print('------NeD------ Best Test : {}'.format(best_test_results))
    with open(os.path.join(args["log_dir"], f'train_{args["seed"]}.log'), 'a+') as f:
        f.write("BEST Epoch: {}\n".format(logger.best_epoch))
        f.write("\t NeD BEST Test: {}\n".format(best_test_results))
    return args

def set_up(args, default_configure):
    args.update(default_configure)
    set_random_seed(args["seed"])
    args["log_dir"] = setup_log_dir(args)
    # print(args)
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser("NetDetect")
    parser.add_argument("-s", "--seed", type=int, default=2022, help="Random seed")
    parser.add_argument("-ld","--log_dir", type=str, default="results", help="Dir for saving training results")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--valid_epoch", type=int, default=1, help="valid epoch")
    parser.add_argument("--MAX_POS_NEG_RATE", type=int, default=1, help="max positive-negative rate")
    parser.add_argument("--MAX_SEQ", type=str, default='[0]', help="max sequence length")
    parser.add_argument('--use_kge', type=bool, default=False, help='encode node simply by TransD')
    parser.add_argument('--use_gat', type=bool, default=False, help='encode node simply by GAT')
    # parser.add_argument('--use_SemAttn', type=bool, default=False, help='encode node simply by GAT')
    parser.add_argument('--use_metapath', type=bool, default=True, help='encode node by GAN')
    parser.add_argument('--use_conven_feat', type=bool, default=True, help='use conventional features')
    parser.add_argument('--path', type=str, default='total', help='include which path')

    default_configure = {
        "lr": 5e-5,
        "num_heads": [2, 2],  # Number of attention heads for node-level attention
        # "hidden_units": 32,
        "dropout": 0.0,
        "weight_decay": 0.001,
        "num_epochs": 30,
        "device":"cuda:1" if torch.cuda.is_available() else "cpu",
        "patience": 10, # 用于判断 early stop
        "init_checkpoint": None, # 一开始是否要加载模型
        "rnn_hidden": 512,
        "node_emb_size": 64,
    }
    args = parser.parse_args().__dict__
    args['model'] = generate_model_name(args)
    args = set_up(args, default_configure=default_configure)

    '''
    model:
        numeric: 仅t时刻，特征为 fin + nonfin 
        basic: 仅t时刻，特征为 fin + nonfin + mda_bert，合起来作为 conventional 特征
        gat, kge, han: encoding node 的方法
    '''
    main(args)
