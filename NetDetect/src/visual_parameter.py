import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime

plt.rcParams['font.sans-serif']=['Times New Roman']

class Visual(object):
    def __init__(self, filelist, X_ticks):
        self.filelist = filelist
        self.X_ticks = X_ticks
        self.plt = self.canvas_init()
        self.set_plot_config()
        self.draw()
        self.save_to_png()

    def canvas_init(self):
        # 修改全局字体
        plt.rc('font', family='Times New Roman', size=20)
        # 设置字体属性字典
        # fontdict = {'family': 'Times New Roman', 'size': 12}

        # 切换绘图后端（backend）。它用于在非交互式环境中生成图像文件，而不是直接显示图形窗口。
        plt.switch_backend('Agg')
        plt.figure(1)
        plt.grid(linestyle='-.', axis='both')

        plt.xticks(self.X_ticks, self.X_ticks)
        # plt.title(args.data_source, fontdict=fontdict)
        # plt.xlabel("Embedding size ${F}$ (for static network embedding)")
        plt.xlabel("Embedding size ${F'}$ (for temporal embedding)")
        # plt.xlabel("Attention head ${H}$")
        # plt.xlabel(r'Propagation iterations ${P}$')

        plt.ylabel("AUC")

        return plt

    def set_plot_config(self):

        self.color = [ 'coral', 'lightskyblue', 'mediumpurple', 'moccasin', 'orchid', 'plum', 'coral', 'gold', 'turquoise',
                 'palegreen', 'skyblue']
        self.marker = ['o', 'x', '+', '.', ',', 'v', 's', '^', '<', '>', '*', '2', '3', '4', '1', 'p', 'h', 'H', 'D', 'd',
                  '|', '_', '.', ',', 'dr']
        self.markersize = 9
        # self.markevery = 1
        # width = 0.12 # 柱状图的宽度


    def read_log(self, filename):
        with open( f'results/{filename}/result.json','r') as f:
            result = json.load(f)
        auc = result['result_avg']['auc']
        fr_prec = result['result_avg']['fr_prec']
        fr_re = result['result_avg']['fr_re']
        le_prec = result['result_avg']['le_prec']
        le_re = result['result_avg']['le_re']
        mac_f1 = result['result_avg']['mac_f1']
        return auc, fr_prec, fr_re, le_prec, le_re, mac_f1

    def draw(self):
        aucs  = []
        for i in range(len(self.filelist)):
            auc, _, _, _, _, _ = self.read_log(self.filelist[i])
            aucs.append(auc)

        self.plt.figure(1)
        self.plt.plot(self.X_ticks, aucs, self.marker[0]+"-", color=self.color[0])

        # legend: 折线的注释
        # self.plt.legend(line_names, loc='upper left')
        self.plt.xlim(0.5, np.max(self.X_ticks) + 0.5)
        self.plt.ylim(0.688, 0.73)

    def save_to_png(self):
        date = str(datetime.datetime.now().date())
        self.plt.figure(1)
        self.plt.savefig(f"images/auc_gru_dim_{date}.png", bbox_inches='tight', transparent=False, dpi=1000, figsize=(50, 50))
        # self.plt.savefig(f"images/auc_attn_heads_{date}.png", bbox_inches='tight', transparent=False, dpi=1000, figsize=(50, 50))
        # self.plt.savefig(f"images/auc_node_dim_{date}.png", bbox_inches='tight', transparent=False, dpi=1000, figsize=(50, 50))
        # self.plt.savefig(f"images/auc_prop_iter_{date}.png", bbox_inches='tight', transparent=False, dpi=1000, figsize=(50, 50))

if __name__ == '__main__':
    attn_heads = [
        'base-han_5e-05_0.0_0.001_[1, 1]_256_30_[]_512_64_None_2023-06-13',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_64_None_2023-06-13',
        'base-han_5e-05_0.0_0.001_[4, 4]_256_30_[]_512_64_None_2023-06-13',
        'base-han_5e-05_0.0_0.001_[8, 8]_256_30_[]_512_64_None_2023-06-13'
    ]
    prop_iter = [
        'base-han_5e-05_0.0_0.001_[2]_256_30_[]_512_64_None_2023-06-19',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_64_None_2023-06-13',
        'base-han_5e-05_0.0_0.001_[2, 2, 2]_256_30_[]_512_64_None_2023-06-19',
        'base-han_5e-05_0.0_0.001_[2, 2, 2, 2]_256_30_[]_512_64_None_2023-06-19',
    ]
    gru_output_dim = [
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_64_64_None_2023-06-24',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_128_64_None_2023-06-11',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_256_64_None_2023-06-11',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_64_None_2023-06-13',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_1024_64_None_2023-06-11'
    ]
    graph_emb_dim = [
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_32_None_2023-06-24',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_64_None_2023-06-13',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_128_None_2023-06-24',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_256_None_2023-06-24'
    ]

    # X_ticks = [1, 2, 3, 4]  # prop iter
    # X_ticks = [1, 2, 4, 8] # attn heads
    X_ticks = [64, 128, 256, 512, 1024] # gru ouput dim
    # X_ticks = [32, 64, 128, 256] # node emb dim
    v = Visual(gru_output_dim, X_ticks)