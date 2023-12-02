import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl import add_self_loop
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, roc_auc_score
from focal_loss import FocalLoss
from attn_sum import AttnSum3d, AttnSum2d
import json
import my_utils

class SemanticAttention(nn.Module):
    """
    就是自注意力加和
    """
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        expand_beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (expand_beta * z).sum(1), beta  # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer. Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    layer_num_heads : number of attention heads
    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features
    Outputs
    -------
        The output feature
    """

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):

        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.LN = nn.LayerNorm(in_size)
        self.BN = nn.BatchNorm1d(layer_num_heads)

        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_feats = in_size,
                    out_feats = out_size,
                    num_heads = layer_num_heads,
                    feat_drop = dropout,
                    attn_drop = dropout,
                    activation = F.elu,
                    allow_zero_in_degree = False, # 是否允许有度为0的点
                )
            )
        # self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        # 可以替代 SemanticAttention
        self.graph_sum = AttnSum3d(dim=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            # i = 1, 2, 3,...10, 10个 meta-path 的图
            # 一个gat_layer 是一个 GAT 网络
            # 给邻接矩阵 g 加上一个对角矩阵，让每个节点和自己相连，否则 GATConv 不允许有度（degree）为 0 的点
            g = add_self_loop(g)
            h = self.LN(h)
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)
        semantic_embeddings = self.LN(semantic_embeddings)
        # (N, M, D * K) -> (N, 1, D * K) -> (N, D * K)
        # attn_coef: (10,)
        sum_embeddings = self.graph_sum(semantic_embeddings)
        # sum_embeddings, attn_coef = self.semantic_attention(semantic_embeddings)
        return sum_embeddings.squeeze(1)  # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        # num_heads: 多阶邻居
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,)
            )
        self.BN = nn.BatchNorm1d(hidden_size * num_heads[-1])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * num_heads[-1], 2 * num_heads[-1] * hidden_size),
            nn.Linear(2 * num_heads[-1] * hidden_size, out_size),
        )
        # 参数 weight: 可以给每个类指定一个权重。通常在训练数据中不同类别的样本数量差别较大时，可以使用权重来平衡。
        # self.loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5,0.5]))
        self.loss = FocalLoss(gamma=4, alpha=0.25, size_average=True)

    def forward(self, g, h):
        # path_attn = []
        for gnn in self.layers:
            h = gnn(g, h)
            # path_attn.append(attn_coef)
            h = self.BN(h)

        # [2, 4111, 10] -> [1, 4111, 10] -> [4111, 10]
        # path_attn = torch.stack(path_attn, dim=0).mean(0).squeeze(1)
        return h, self.mlp(h)

    def score(self, logits, labels):
        # train 阶段

        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
        labels = labels.cpu().numpy()

        accuracy = (prediction == labels).sum() / len(prediction)
        # micro f1 和 macro f1 是多分类任务里的指标
        # micro f1 是计算出所有类别总的Precision和Recall，然后计算F1，Micro-F1则更容易受到常见类别的影响。
        # macro f1 计算出每一个类的Precison和Recall后计算F1，最后将F1平均，Macro-F1平等地看待各个类别，它的值会受到稀有类别的影响
        micro_f1 = f1_score(labels, prediction, average="micro")
        macro_f1 = f1_score(labels, prediction, average="macro")
        return accuracy, micro_f1, macro_f1

    def evaluate_(self, g, features, labels, mask):
        # valid 阶段
        with torch.no_grad():
            output, logits = self.forward(g, features)

        loss = self.loss(logits[mask], labels[mask])
        accuracy, micro_f1, macro_f1 = self.score(logits[mask], labels[mask])

        return loss, accuracy, micro_f1, macro_f1

class Model(nn.Module):
    def __init__(self, embs, dropout,
                 num_path, out_size, num_heads, args # 这一行的参数全是han的
        ):
        super(Model, self).__init__()
        fin_embs, nonfin_embs, mda_bert_embs, node_embs = embs[0], embs[1], embs[2], embs[3]
        self.n_graph = node_embs.shape[0] # 19 年
        self.n_entities = node_embs.shape[1] # 4111
        # self.in_size = node_embs.shape[2] # 64
        self.in_size = args["node_emb_size"]
        self.rnn_hidden = args["rnn_hidden"] # hidden size of GRUx
        self.n_path = num_path
        self.max_seq_len = 6
        self.dropout = dropout
        self.fin_size = 84 # 财务指标有84个
        self.nfin_size = 11 # 非财务指标有11个
        self.numeric_size = self.fin_size + self.nfin_size # 84 + 11 = 95
        self.args = args
        # padding_idx: 表示用于填充的参数索引，比如用3填充，则行index为3的embedding设置为0
        # 设置成-1就令最后一行 embedding 为 0 向量
        self.fin_embs = nn.Embedding(fin_embs.shape[0] + 1, self.fin_size, padding_idx=-1)
        self.nfin_embs = nn.Embedding(nonfin_embs.shape[0] + 1, self.nfin_size, padding_idx=-1)
        self.bert_embs = nn.Embedding(mda_bert_embs.shape[0] +1, 768, padding_idx=-1)

        self.fin_embs.weight.data[:-1].copy_(fin_embs)
        self.nfin_embs.weight.data[:-1].copy_(nonfin_embs)
        self.bert_embs.weight.data[:-1].copy_(mda_bert_embs)

        # 所有图的embedding拼接起来, nn.Embedding 是一个查询表，只能存储二维向量
        self.node_embs = nn.Embedding(self.n_graph * self.n_entities + 1, self.in_size)
        # 储存HAN输出的节点embedding，用于之后的RNN的查询，shape和node_embs一样
        self.han_outputs = nn.Embedding(self.n_graph * self.n_entities + 1, self.in_size, padding_idx=-1)

        self.USE_GRAPH = True if self.args['use_kge'] or self.args['use_gat'] or self.args['use_metapath'] else False
        # self.IS_SEQ = False if len(json.loads(self.args['MAX_SEQ'])) == 1 else True
        self.IS_SEQ = True

        # 初始化 self.node_embs
        if self.args['use_kge']:
            # self.in_size 必须为64
            # 如果要使用KG embedding的话，就要用 node_embs 初始化 self.node_embs；如果只是随机初始化节点向量，就不需要下面这行操作
            self.node_embs.weight.data[:-1].copy_(node_embs.reshape((-1, self.in_size)))
        else:
            if self.args['emb_init']=='Xavier':
                # Xavier Initialization
                nn.init.xavier_uniform_(self.node_embs.weight)
                # nn.init.xavier_uniform_(self.han_outputs.weight)
            elif self.args['emb_init']=='Normal':
                # Norm Initialization
                nn.init.normal_(self.node_embs.weight, mean=0.0, std=0.1)
                # nn.init.normal_(self.han_outputs.weight, mean=0.0, std=0.1)
            else: # 不初始化
                pass

        if self.args['use_gat']:
            self.gat_LN = nn.LayerNorm(self.in_size)
            self.GATs = nn.ModuleList()
            for i in range(self.n_graph):
                self.GATs.append(GATConv(
                    in_feats = self.in_size, out_feats = int(self.in_size / num_heads[0]),
                    num_heads = num_heads[0], feat_drop = self.dropout, attn_drop = self.dropout, activation = F.elu,
                    allow_zero_in_degree = True, # 是否允许有度为0的点
                ))

        if self.args['use_metapath']:
            # self.path_attns = nn.Embedding(self.n_graph * self.n_entities + 1, self.n_path, padding_idx=-1)
            # self.han_LN = nn.LayerNorm(self.in_size)
            self.HANs = nn.ModuleList()
            for i in range(self.n_graph): # 19 年
                self.HANs.append(
                    HAN(num_meta_paths=self.n_path, in_size = self.in_size, hidden_size = int(self.in_size / num_heads[0]),
                                    out_size = out_size, num_heads = num_heads, dropout=dropout)
                )

        self.mlp_in_dim = self.in_size if self.USE_GRAPH else 0

        if self.args['use_conven_feat']:
            # self.mda_encoder = nn.Sequential(nn.Linear(768, self.in_size),
            #       nn.Linear(self.hidden_size, self.in_size),
            # )
            self.mlp_in_dim = self.mlp_in_dim + self.numeric_size + 768

        # 文档：https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        # Hin: input_size, 输入的特征的维度
        # Hout: hidden_size, hidden state 向量的维度，以及ouput输出向量的维度
        # B: batch_size
        # L: sequence_length
        # D: 2 if bidirectional = True, else False

        # num_layers: 有几层 GRU
        # bidirectional: If True, becomes a bidirectional GRU. Default: False
        # batch_first: True，input为 [B, L, Hin], 否则 [L, B, Hin]
        if self.IS_SEQ:
            self.LN = nn.LayerNorm(self.mlp_in_dim)
            self.rnn = nn.GRU(input_size = self.mlp_in_dim,
                              hidden_size = self.rnn_hidden,
                              # num_layers = -max(json.loads(args["MAX_SEQ"])) + 1, # 往前预测多少期
                              num_layers = 1,
                              dropout = dropout,
                              bidirectional = False,
                              batch_first = True)
            self.mlp_in_dim = self.rnn_hidden
            # self.dense = nn.Linear(self.mlp_in_dim, self.mlp_in_dim)

        self.BN = nn.BatchNorm1d(self.mlp_in_dim, eps=1e-4)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_in_dim, 2 * self.mlp_in_dim),
            nn.Tanh(), # Tanh 会减缓训练的速度，具体会不会让效果变差不好说
            # nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.Linear(2 * self.mlp_in_dim, out_size),
        )

        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5,0.5]))
        # self.loss = FocalLoss(gamma=2, alpha=0.25, size_average=True)

    def gat_infer(self, i, adjs, indices):
        # train gat
        node_embs = self.gat_LN(self.node_embs(indices))
        output = self.GATs[i](adjs, node_embs).flatten(1)

        self.node_embs.weight.data[i*self.n_entities: (i+1)*self.n_entities].copy_(output)

    # 输出hans modules
    def hans_infer(self, i, adjs, indices):
        # 第 i 年，第 i 个图
        # node_embs = self.han_LN(self.node_embs(indices))
        node_embs = self.node_embs(indices)
        output, logits = self.HANs[i](adjs, node_embs)
        # 把跑出来的embedding 存到模型中
        # path_attn: [4111, 10]
        # self.path_attns.weight.data[i * self.n_entities: (i+1)*self.n_entities].copy_(path_attn)

        self.han_outputs.weight.data[i*self.n_entities: (i+1)*self.n_entities].copy_(output)
        return self.HANs[i], logits

    def test_hans(self, i, adjs, labels, mask, indices):
        return self.HANs[i].evaluate_(adjs, self.node_embs(indices), labels, mask)

    def forward(self, node_seq, fin_seq, nfin_seq, mda_seq, seq_len):
        inputs = []
        if self.args['use_metapath']:
            # [256, 6, 10]
            # node_path_attn = self.path_attns[node_seq]
            # node_emb = self.node_embs(node_seq)
            inputs.append(self.han_outputs(node_seq))
        elif self.args['use_kge']:
            inputs.append(self.node_embs(node_seq))
            # node_emb = self.node_embs(node_seq)
        elif self.args['use_gat']:
            inputs.append(self.node_embs(node_seq))
            # node_emb = self.node_embs(node_seq)

        # input = torch.cat([node_emb], dim=-1) if self.USE_GRAPH else torch.tensor([])

        if self.args['use_conven_feat']:
            fin_emb = self.fin_embs(fin_seq)
            nfin_emb = self.nfin_embs(nfin_seq)
            bert_emb = self.bert_embs(mda_seq)

            # 改变维度
            # low_dim_bert_emb = self.mda_encoder(bert_emb)
            # [B, 1, H] -> [B, H], 如果是 [B, 2, H]，就不会squeeze
            # output = torch.cat([input, fin_emb, nfin_emb, bert_emb], dim=-1).squeeze(1)
            inputs += [fin_emb, nfin_emb, bert_emb]
            # input = torch.cat([input, fin_emb, nfin_emb, bert_emb], dim=-1)

        output = torch.cat(inputs, dim=-1)

        if self.IS_SEQ:
            # rnn(input, h0): h0是 GRU 的hidden statede的初始向量，不设的话默认是全为0的embedding
            # input: [B, len, H]
            # h0: [D * num_layers, B, Hout] 如果bidirectional为True的话，D=2
            output = self.LN(output)
            gru_output, _ = self.rnn(output)
            # 返回 output, h_n
            #   output: [B, len, D * Hout]
            #   h_n: [D * num_layers, B, Hout]
            # 把 padding 的部分去掉
            # output = self.dense(gru_output)
            output = self.gather_indexes(gru_output, seq_len-1) # [B H]
        else:
            output = output.squeeze(1)

        output = self.BN(output)
        logits = self.mlp(output)
        return logits, output

    def gather_indexes(self, output, gather_index):
        # 参考 recbole 里的 GRU4Rec 用法
        """Gathers the vectors at the specific positions over a minibatch"""
        # torch.expand: 参数为传入指定shape，在原shape数据上进行高维拓维，根据维度值进行重复赋值。
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        # dim=1 表示按行号进行索引
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def score(self, logits, labels, loss=None):
        # train 阶段
        # logits本来shape是[-1, 2]，2为类别数，现在取最大那一类的概率作为最后的预测值，转换为 [-1]
        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
        labels = labels.cpu().numpy()

        auc, acc, fr_prec, fr_recall, le_prec, le_recall, macro_f1, micro_f1, pos_neg_rate = my_utils.evaluate(
            prediction, labels)

        output_dict = {
            'loss': round(loss, 4),
            'auc': auc, 'fr_prec': fr_prec, 'fr_re': fr_recall,
            'le_prec': le_prec, 'le_re': le_recall,
            'mac_f1': macro_f1, 'mic_f1': micro_f1,
            'acc': acc, 'pos_neg': pos_neg_rate
        }
        return output_dict

    def evaluate_(self, node_seq, fin_seq, nfin_seq, mda_seq,  seq_len, labels):
        # valid 阶段
        with torch.no_grad():
            logits, output = self.forward(node_seq, fin_seq, nfin_seq, mda_seq, seq_len)

        loss = self.loss(logits, labels)

        output_dict = self.score(logits, labels, loss.item())

        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()

        return output_dict, output.cpu().numpy()
