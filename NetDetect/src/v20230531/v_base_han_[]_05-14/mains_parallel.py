import main
import numpy as np
import my_utils
from my_utils import EarlyStopping, set_random_seed, setup_log_dir
import os
import json
import argparse
import torch
import re
import collections

def run_n_fold(seeds=[2021, 2022, 2023, 2024, 2025]):
    '''设定不同seed，重复运行main.py中main函数，并把结果分别存起来'''
    for seed in seeds:
        print('-' * 60 + f' seed:{seed} ' + '-' * 60)
        args["seed"] = seed
        set_random_seed(args["seed"])
        main.main(args)


def calculate_avg_results(save_path):
    results = collections.defaultdict(list)
    for root, dirs, files in os.walk(save_path):

        # 只获取.log文件，其他.json文件不会读进来
        log_files = [f for f in files if '.log' in f]
        for log in log_files:
            with open(os.path.join(save_path, log), 'r') as f:
                output = f.readlines()[-1].replace('NeD BEST Test:', '')
            output = re.sub(r'\t| ', '', output)
            output = re.sub(r"'", r'"', output)
            output = json.loads(output)
            results['auc'].append(output['auc'])
            results['fr_prec'].append(output['fr_prec'])
            results['fr_re'].append(output['fr_re'])
            results['le_prec'].append(output['le_prec'])
            results['le_re'].append(output['le_re'])
            results['mac_f1'].append(output['mac_f1'])

    with open(os.path.join(save_path,'result.json'), 'w') as f:
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

def set_up(args, default_configure):
    args.update(default_configure)
    args["log_dir"] = setup_log_dir(args)
    # print(args)
    return args

if __name__=='__main__':
    parser = argparse.ArgumentParser("NetDetect")
    # parser.add_argument("-s", "--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("-ld", "--log_dir", type=str, default="results", help="Dir for saving training results")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--valid_epoch", type=int, default=1, help="valid epoch")
    parser.add_argument("--MAX_POS_NEG_RATE", type=int, default=1, help="max positive-negative rate")
    parser.add_argument("--MAX_SEQ", type=str, default='[]', help="max sequence length")
    parser.add_argument('--use_kge', type=bool, default=False, help='encode node simply by TransD')
    parser.add_argument('--use_gat', type=bool, default=False, help='encode node simply by GAT')
    # parser.add_argument('--use_SemAttn', type=bool, default=False, help='encode node simply by GAT')
    parser.add_argument('--use_metapath', type=bool, default=True, help='encode node by GAN')
    parser.add_argument('--use_conven_feat', type=bool, default=True, help='use conventional features')
    parser.add_argument('--path', type=str, default='total', help='include which path')
    parser.add_argument('--emb_init', type=str, default='None', help='only gat and han will use')

    default_configure = {
        "lr": 5e-5,
        "num_heads": [2, 2],  # Number of attention heads for node-level attention
        # "hidden_units": 32,
        "dropout": 0.0,
        "weight_decay": 0.001,
        "num_epochs": 30,
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "patience": 10,  # 用于判断 early stop
        "init_checkpoint": None,  # 一开始是否要加载模型
        "rnn_hidden": 512,
        "node_emb_size": 64,
    }
    args = parser.parse_args().__dict__
    args['model'] = main.generate_model_name(args)
    # set_up 先不指定 seed
    args = set_up(args, default_configure=default_configure)

    '''
    model:
        numeric: 仅t时刻，特征为 fin + nonfin 
        basic: 仅t时刻，特征为 fin + nonfin + mda_bert，合起来作为 conventional 特征
        gat, kge, han: encoding node 的方法
    '''
    seeds = [2023]
    # seeds = [2023, 2021, 2022, 2024, 2025, 2026, 2027, 2028, 2029, 2020]
    run_n_fold(seeds=seeds)

    # save_path = args["log_dir"]
    save_path = 'results/' + 'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_64_None_2023-05-14'
    # calculate_avg_results(save_path=save_path)