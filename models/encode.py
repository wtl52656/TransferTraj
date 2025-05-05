import math

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)


class Regressor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor, self).__init__()
        self.linear_wide = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_deep = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_recurrent = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer = MultiLayerPerceptron(output_dim, (output_dim,), output_layer=True)

    def forward(self, deep, recurrent):
        fuse = self.linear_deep(deep) + self.linear_recurrent(recurrent)
        return self.out_layer(fuse)

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)


    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)

        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices, F.softmax(logits, dim=-1)

class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embd):
        super().__init__()
        dropout = 0.1
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts=8, top_k=4):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices, softmax_gating_output = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output, softmax_gating_output

class PositionalEncode(nn.Module):
    """Non-learnable positional encoding layer proposed in the Transformer.
    """

    def __init__(self, hidden_size):
        super(PositionalEncode, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        B, L = pos_seq.shape
        sinusoid_inp = torch.ger(rearrange(pos_seq, 'B L -> (B L)'), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = rearrange(pos_emb, '(B L) E -> B L E', B=B, L=L)

        return pos_emb


class FourierEncode(nn.Module):
    """A type of trigonometric encoding for encode continuous values into distance-sensitive vectors.
    """

    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        Args:
            x (Tensor): input sequence for encoding, (batch_size, seq_len, 1)

        Returns:
            Tensor: encoded sequence, shape (batch_size, seq_len, embed_size)
        """
        if len(x.shape) < 3:
            x = x.unsqueeze(-1)

        encode = x * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode



# 以小数为索引
class RoPE_Attention_float(nn.Module):
    def __init__(self, hidd_dim):
        super(RoPE_Attention_float, self).__init__()
        
        self.hidd_dim = hidd_dim

        # 这三个是注意力的KQV特征变换
        self.wq = nn.Linear(self.hidd_dim, self.hidd_dim)
        self.wk = nn.Linear(self.hidd_dim, self.hidd_dim)
        self.wv = nn.Linear(self.hidd_dim, self.hidd_dim)
        
        # self.Wr_lng = nn.Linear(1, self.hidd_dim // 2, bias=False)
        # self.Wr_lat = nn.Linear(1, self.hidd_dim // 2, bias=False)
        self.Wr = nn.Linear(2, self.hidd_dim // 2, bias=False)
    def forward(self, x, norm_coord, causal_mask, batch_mask):
        # 先进性特征变换
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 使用旋转位置编码，这里不是真实的下标，而是x的位置

        # attention 操作之前，应用旋转位置编码
        
        xq, xk = self.apply_rotary_emb(xq, xk, norm_coord)

        #计算因果注意力，并mask填充部分
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.hidd_dim)
        
        scores = scores.masked_fill(causal_mask, float('-inf'))
        scores = scores.masked_fill(batch_mask, float('-inf'))
        # print(scores)
        # exit()
        scores = F.softmax(scores.float(), dim=-1)
        
        # mask操作计算
        # scores = scores.masked_fill(batch_mask, 0)
        output = torch.matmul(scores, xv)

        return output

    # 旋转位置编码计算
    def apply_rotary_emb(self, xq, xk, norm_coord):
        # xq.shape = [batch_size, seq_len, dim]
        # xq_.shape = [batch_size, seq_len, dim // 2, 2]

        _, traj_len, _ = xq.shape

        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
        
        
        # print(xq_.shape)
        # 转为复数域
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)
        
        # 根据x计算他对应的旋转位置，但x有两列，对每一列分别进行位置的编码
        _freqs_cis = self.precompute_freqs_cis(norm_coord)
        
        
        # 输入一个下标推算对应的结果->输入任意的一个值（偏移+缩放之后的结果）
        # 系数变为可学习参数


        # 应用旋转操作，然后将结果转回实数域
        # xq_out.shape = [batch_size, seq_len, dim]
        xq_out = torch.view_as_real(xq_ * _freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * _freqs_cis).flatten(2)
        
        return xq_out.type_as(xq), xk_out.type_as(xk)


    # 生成旋转矩阵
    def precompute_freqs_cis(self, norm_coord):
        freqs = self.Wr(norm_coord)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
        return freqs_cis





class RAFEE_Encoder(nn.Module):
    def __init__(self, dim, layers, max_seq_len = 10000):
        super().__init__()
        self.dim = dim

        #  max_seq_len设置成可学习的参数

        self.max_seq_len = max_seq_len #轨迹的最大长度

        self.layers = nn.ModuleList([RAFEE_Encoder_layer(dim) 
                                     for _ in range(layers)])
        
    
    def forward(self, x, norm_coord, mask, src_key_padding_mask):
        _, max_seq_len, _ = x.shape

        src_key_padding_mask = src_key_padding_mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, norm_coord, mask, src_key_padding_mask)
        return x

class RAFEE_Encoder_layer(nn.Module):
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()

        self.dim = dim

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        
        # self.freqs_cis = 
        

        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.MoE_fc = SparseMoE(dim)
        
        self.fc_norm = nn.LayerNorm(dim)
        self.RoPE_Attention_float = RoPE_Attention_float(dim)

    def forward(self, x, norm_coord, causal_mask, batch_mask):
        x_attn = self.RoPE_Attention_float(x, norm_coord, causal_mask, batch_mask)
        x = x + x_attn
        x = self.norm(x)

        # x_fc = self.fc(x)
        final_output, softmax_gating_output = self.MoE_fc(x)
        # print(x.shape, final_output.shape)
        # exit()
        x = x + final_output
        x = self.fc_norm(x)
        return x
