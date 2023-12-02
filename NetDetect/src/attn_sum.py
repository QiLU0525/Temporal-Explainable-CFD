import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class AttnSum3d(nn.Module):
    def __init__(self,
                 dim,
                 TRAINABLE=False):
        super(AttnSum3d, self).__init__()
        # AttnSum 可以把有 padding 的 [m*n] 矩阵不用知道 m 的值就进行压缩
        self.dim = dim
        self.TRAINABLE = TRAINABLE
        if self.TRAINABLE:
            # 这是带权重矩阵的，会给Q、K、V前面乘权重矩阵，最后产生的结果会不一样
            self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=1)

    def forward(self, input, mask=None):
        # input : [b, m, dim]
        # mask: [b, m], eg:
        #    [[1,1,1,0],
        #     [1,1,0,0],
        #     [1,0,0,0]]
        # out: [b, 1, dim]

        if self.TRAINABLE:
            # Q, K, V
            out, attn_weights = self.attn(input, input, input)
            out = out.mean(dim=1)
        else:
            # 新attn：T-0效果会变差；先对 attn 权重进行平均，让 attn 为 (4111, 25, 1)
            attn_weights = torch.bmm(input, input.permute(0, 2, 1)).mean(2, keepdim=True)

            if mask is not None:
                # 将 mask 中为0的部分替换为 -inf, masked_fill 是把 1 的地方替换为 -inf，所以用 1-mask
                # mask.unsqueeze(-1): [4111, 25, 1]
                attn_weights = attn_weights.masked_fill((1 - mask.unsqueeze(-1)).bool(), float('-inf'))

                norm_attn_weights = F.softmax(attn_weights, dim=1)
                # softmax之后，再 mask 一次，目的是将有些所有值为 nan 的行置为值全为0
                norm_attn_weights = norm_attn_weights.masked_fill((1 - mask.unsqueeze(-1)).bool(), 0.0)
            else:
                norm_attn_weights = F.softmax(attn_weights, dim=1)

            # (4111, 1, 25) x (4111, 25, 64) -> (4111, 1, 64)
            out = torch.bmm(norm_attn_weights.permute(0, 2, 1), input)
            '''
            # 老的attn，在做T-0的时候效果好，有时序的时候效果不好
            # attn 权重就是 (4111, 25, 25)，乘完 emb 以后再进行平均
            attn_weights = torch.bmm(input, input.permute(0, 2, 1))
            if mask is not None:
                attn_weights = attn_weights.masked_fill(1 - mask.unsqueeze(-1), float('-inf'))
            norm_attn_weights = torch.softmax(attn_weights, dim=1)
            out = torch.bmm(norm_attn_weights.permute(0, 2, 1), input).mean(dim=1, keepdim=True)
            '''
        return out, norm_attn_weights

class AttnSum2d(nn.Module):
    def __init__(self,
                 dim,
                 TRAINABLE=False):
        super(AttnSum2d, self).__init__()
        # AttnSum 可以把有 padding 的 [m*n] 矩阵不用知道 m 的值就进行压缩
        self.dim = dim
        self.TRAINABLE = TRAINABLE
        if self.TRAINABLE:
            # 这是带权重矩阵的，会给Q、K、V前面乘权重矩阵，最后产生的结果会不一样
            self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=1)

    def forward(self, input):
        # input : [m, dim]
        # out: [1, dim]
        if self.TRAINABLE:
            # Q, K, V
            out, attn_weights = self.attn(input, input, input)
            out = out.mean(dim=0)
        else:
            # [m, m] <- [m, dim] * [dim, m]
            attn_weights = torch.mm(input, input.permute(1, 0))
            attn_weights = torch.softmax(attn_weights, dim=1)
            # (m, m) x (m, dim) -> (m, dim) -> (1, dim)
            # out = torch.mm(attn_weights.mean(1).unsqueeze(0), input)
            # 如果有 batch 的话就把 torch.mm 改成 torch.bmm
            out = torch.mm(attn_weights.permute(1, 0), input).mean(dim=0, keepdim=True)
        return out
