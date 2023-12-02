import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime
# plt 版本不一样，有的版本没有 Times New Roman，装 v3.1.0
plt.rcParams['font.sans-serif']=['Times New Roman']

class Visual(object):
    def __init__(self, filelist, model_list):
        self.filelist = filelist
        self.model_list = model_list
        self.plt = self.canvas_init()
        self.set_plot_config()
        self.draw()
        self.save_to_png()

    def canvas_init(self):
        # 修改全局字体
        plt.rc('font', family='Times New Roman', size=11)
        # 设置字体属性字典
        # fontdict = {'family': 'Times New Roman', 'size': 12}

        # 切换绘图后端（backend）。它用于在非交互式环境中生成图像文件，而不是直接显示图形窗口。
        plt.switch_backend('Agg')
        plt.figure(1)
        plt.grid(linestyle='-.', axis='y')

        plt.xticks(np.arange(1, len(model_list)+1), model_list)
        # plt.title(args.data_source, fontdict=fontdict)
        # plt.xlabel("NetDetect with different timespan")
        plt.xlabel("NetDetect with different early warning")
        # plt.ylabel("value")

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
        line_names = [
            'AUC',
            'Fraud recall (Type I error)',
            'Legit recall (Type II error)',
        ]
        aucs, fr_res, le_res = [], [], []
        for i in range(len(self.filelist)):
            auc, _, fr_re, _, le_re, _ = self.read_log(self.filelist[i])
            aucs.append(auc)
            fr_res.append(fr_re)
            le_res.append(le_re)

        self.plt.figure(1)
        self.plt.plot(range(1, len(aucs)+1), aucs, self.marker[0]+"-", color=self.color[0])
        self.plt.plot(range(1, len(fr_res) + 1), fr_res, self.marker[1]+"-", color=self.color[1])
        self.plt.plot(range(1, len(le_res) + 1), le_res, self.marker[2]+"-", color=self.color[2])
        # legend: 折线的注释
        self.plt.legend(line_names, loc='upper left')
        self.plt.ylim(0.62, 0.81)

    def save_to_png(self):
        date = str(datetime.datetime.now().date())
        self.plt.figure(1)
        # self.plt.savefig(f"images/time_span_{date}.png", bbox_inches='tight', transparent=False, dpi=600, figsize=(50, 50))
        self.plt.savefig(f"images/early_warning_{date}.png", bbox_inches='tight', transparent=False, dpi=600,
                         figsize=(50, 50))


if __name__ == '__main__':

    '''file_list = [
        'base-han_5e-05_0.0_0.001_[2, 2]_512_30_[0]_512_64_Xavier_2023-05-13',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[-1,0]_512_64_None_2023-05-25',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[-2,-1,0]_512_64_None_2023-05-25',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[-3,-2,-1,0]_512_64_None_2023-05-25',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[-4,-3,-2,-1,0]_512_64_None_2023-05-25',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_64_None_2023-05-18'
    ]'''
    file_list = [
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[-5]_512_64_None_2023-05-25',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[-5,-4]_512_64_None_2023-05-25',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[-5,-4,-3]_512_64_None_2023-05-25',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[-5,-4,-3,-2]_512_64_None_2023-05-25',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[-5,-4,-3,-2,-1]_512_64_None_2023-05-25',
        'base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_64_None_2023-05-18'
    ]

    # model_list = ['T-0', 'T-1', 'T-2', 'T-3', 'T-4', 'T-5']
    model_list = ['T-5', 'T-5~T-4', 'T-5~T-3', 'T-5~T-2', 'T-5~T-1', 'T-5~T-0']
    v = Visual(file_list, model_list)