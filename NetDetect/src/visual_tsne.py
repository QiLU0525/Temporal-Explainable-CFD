import torch
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import axes3d
import datetime
# matplotlib 版本: 3.1.0

class Visual(object):
    def __init__(self, filepath, params, dim=2, IS_TUNE=True):
        self.filepath = filepath
        self.dim = dim

        self.plt = self.canvas_init()
        self.load_data(filepath)
        self.params = self.opt_tune() if IS_TUNE else params

        self.tsne = TSNE(n_components=self.dim, perplexity=self.params['perplexity'],
                         learning_rate=self.params['learning_rate'],
                         init=self.params['init'])

        self.draw()
        self.save_to_png()

    def canvas_init(self):
        # 修改全局字体
        plt.rc('font', family='Times New Roman', size=11)
        fontdict = {'family': 'Times New Roman', 'size': 12}
        # 切换绘图后端（backend）。它用于在非交互式环境中生成图像文件，而不是直接显示图形窗口。
        plt.switch_backend('Agg')
        plt.figure(1)
        plt.grid(linestyle='-.', axis='y')

        # plt.xticks(np.arange(1, len(model_list)+1), model_list)
        plt.title('Visualization embedding', fontdict=fontdict)
        # plt.xlabel("NetDetect with different timespan")
        # plt.xlabel("NetDetect with different early warning")
        # plt.ylabel("value")
        return plt

    def load_data(self, filepath):
        corp_data = torch.load('ChinaCorp_1x.pt')
        test_idx = corp_data['split_idx']['test']
        self.Y_true_test = corp_data['label'][test_idx].numpy()
        self.X = np.load(filepath)

    def tune_objective(self, trial):
        # 需要 grid search 的参数
        init = trial.suggest_categorical('init', ['random', 'pca'])
        perplexity = trial.suggest_int('perplexity', 5, 50)
        learning_rate = trial.suggest_float('learning_rate', 50, 500)
        # 创建临时 t-SNE 模型
        tsne = TSNE(n_components=self.dim, perplexity=perplexity, learning_rate=learning_rate, init=init)
        # 转换数据
        X_tsne = tsne.fit_transform(self.X)
        # 计算聚类效果评分，这里可以根据具体需求定义自己的评分指标
        score = silhouette_score(X_tsne, self.Y_true_test)

        return score

    def opt_tune(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.tune_objective, n_trials=50)
        best_params = study.best_params
        best_score = study.best_value
        print(f'最好的参数是: {best_params}')
        return best_params

    def draw(self):
        fig = self.plt.figure(1)
        X_tsne = self.tsne.fit_transform(self.X)
        if self.dim == 3:
            ax = axes3d.Axes3D(fig)
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c= self.Y_true_test)
        else:
            self.plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.Y_true_test)

    def save_to_png(self):
        date = str(datetime.datetime.now().date())
        self.plt.figure(1)
        self.plt.savefig(f"images/tsne_2d_{date}.png", bbox_inches='tight', transparent=False, dpi=600,
                         figsize=(50, 50))

if __name__=="__main__":
    filepath = 'results/base-han_5e-05_0.0_0.001_[2, 2]_256_30_[]_512_64_None_2023-05-25/output_2020.npy'
    params = {
        'init': 'pca',
        'perplexity': 200,
        'learning_rate': 300,
    }
    Visual(filepath, params, dim = 2)
