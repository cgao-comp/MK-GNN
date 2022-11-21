import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse


from torch.nn.parameter import Parameter

class dot_attention(nn.Module):
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: 上下文张量和attention张量。
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和v做点积。
        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):
    """ 多头自注意力"""
    def __init__(self, model_dim=64, num_heads=4, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads   # 每个头的维度
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)         # LayerNorm 归一化。

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # 线性映射。
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 按照头进行分割
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # 缩放点击注意力机制
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # 进行头合并 concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # 进行线性映射
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # 添加残差层和正则化层。
        output = self.layer_norm(residual + output)

        return output, attention

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = torch.sparse_tensor_dense_matmul(x, y)
    else:
        res = torch.matmul(x, y)
    return res

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, voc_size, emb_dim, device=torch.device('cpu:0')):
        super(SGC, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        # self.x = torch.eye(voc_size).to(device)
        self.W = nn.Linear(voc_size, emb_dim)

    def forward(self, input):
        #print(self.x)
        return self.W(input)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features #145
        self.out_features = out_features #64
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  #145 * 64
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  #64
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
