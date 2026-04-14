import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import os

from layer import MLP, Propagation
from gsl_uu import GSL4uu
from utils import *

class   Encoder(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Encoder, self).__init__()
        if args.dataset in ['yelp1']:
            self.user_emb_ego = torch.nn.Embedding(
                num_embeddings=num_users, embedding_dim=args.hidden_dim)
            self.item_emb_ego = torch.nn.Embedding(
                num_embeddings=num_items, embedding_dim=args.hidden_dim)
            # nn.init.normal_(self.user_emb_ego.weight, std=0.1)
            # nn.init.normal_(self.item_emb_ego.weight, std=0.1)
            torch.nn.init.xavier_normal_(self.user_emb_ego.weight)
            torch.nn.init.xavier_normal_(self.item_emb_ego.weight)
        else:
            self.user_map = MLP(args.user_feat_dim, args.hidden_dim, args.dropout)
            self.item_map = MLP(args.item_feat_dim, args.hidden_dim, args.dropout)

        self.prop_ui = Propagation(args.n_layers, args.dropout_adj) # u-i图卷积
        self.prop_uu = Propagation(args.n_layers, args.dropout_adj)  # u-u图卷积
        self.user_emb_map = nn.Linear(2*args.hidden_dim, args.hidden_dim) 
        self.num_users = num_users
        self.num_items = num_items

    
    def forward(self, ui_graph, uu_graph, user_feat, item_feat, train=True, temp_flag=False):
        if user_feat is None:
            # 无初始用户特征，使用随机初始化的embedding
            user_emb_ego = self.user_emb_ego.weight
            item_emb_ego = self.item_emb_ego.weight
        else:
            # 有用户初始特征，使用MLP得到用户embedding
            user_emb_ego = self.user_map(user_feat)
            item_emb_ego = self.item_map(item_feat)

        
        item_emb = item_emb_ego
        
        all_emb = torch.cat([user_emb_ego, item_emb_ego])
        all_emb = self.prop_ui(ui_graph, all_emb, train)
        user_ui_emb, item_emb = torch.split(all_emb, [self.num_users, self.num_items]) 

        # user_uu_emb = self.prop_uu(uu_graph, F.normalize(user_ui_emb), train)
        user_uu_emb = self.prop_uu(uu_graph, user_emb_ego, train)
        user_final_emb = user_uu_emb
        # user_final_emb = self.user_emb_map(torch.cat([user_uu_emb, user_ui_emb], -1)) # 用户自身emb，二部图用户emb，social图用户emb

        if train:
            return user_uu_emb, user_ui_emb, item_emb, user_emb_ego, item_emb_ego
        else:
            if temp_flag:
                return user_final_emb, item_emb, user_emb_ego
            else:
                return user_final_emb, item_emb
            
'''
class Encoder2(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Encoder2, self).__init__()
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = args.hidden_dim

        self.p_dims = [self.hidden_dim, self.num_items]
        self.q_dims = [self.num_items+self.num_users, self.hidden_dim]
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]  # [I,128]
        self.drop = nn.Dropout(args.dropout_adj)
        self.sigmoid = nn.Sigmoid()

        self.attention_dense = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )

        # self.s_layers = nn.ModuleList([nn.Linear(self.num_users, self.hidden_dim * 2)])
        self.s_layers = nn.ModuleList([
            nn.Linear(self.num_users, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim*2)
        ])

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])

        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
     
        self.s_p_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.num_users)])


    def encode(self, ui_graph):
        for i, layer in enumerate(self.q_layers):
            if i == 0:
                h = layer(ui_graph)  # [U+I,U+I] *[U+I,128] = [U+I,128]
            else:
                embs = layer(torch.sparse.mm(ui_graph, h))  # sparse[U+I,U+I] * [U+I,128] = [U+I,128]
                _, h = torch.split(embs, [self.num_users, self.num_items])  # _:[U,128], h:[I,128]
            h = self.drop(h)  # Dropout
            
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)  # 非最后一层应用 tanh
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
                return mu, logvar
    
    def social_encode(self, uu_graph):        
        for i, layer in enumerate(self.s_layers):
            if i == 0:
                h = layer(uu_graph)
            else:
                h = layer(torch.sparse.mm(uu_graph, h))  # [U,U] * [U,64] = [U,64]
            h = self.drop(h)

            if i != len(self.s_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.hidden_dim]
                logvar = h[:, self.hidden_dim:]
                return mu, logvar
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return self.sigmoid(h)

    def social_decode(self, z):
        h = z
        for i, layer in enumerate(self.s_p_layers):
            h = layer(h)
            if i != len(self.s_p_layers) - 1:
                h = torch.tanh(h)
        return self.sigmoid(h)

    def reparameterize(self, mu, logvar):
        # 重新参数化
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
        else: 
            z = mu + std
        return z
'''


'''
class inter_denoise_gate(torch.nn.Module):
    def __init__(self, args):
        super(inter_denoise_gate, self).__init__()
        # 门控权重生成层
        self.W_gate_cf = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.W_gate_soc = nn.Linear(args.hidden_dim, args.hidden_dim)
        
        # 跨视图交互层
        self.W_fusion = nn.Linear(2 * args.hidden_dim, args.hidden_dim)
        
        # 归一化与激活
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.layer_norm = nn.LayerNorm(args.hidden_dim)  # 稳定训练

    def forward(self, embs_cf, embs_soc):
        # 步骤1：生成动态门控权重
        gate_cf = self.sig(self.W_gate_cf(embs_cf))  # 点击视图重要性 [batch, dim]
        gate_soc = self.sig(self.W_gate_soc(embs_soc))  # 社交视图重要性 [batch, dim]
        
        # 步骤2：跨视图特征交互
        combined = torch.cat([embs_cf * gate_cf, embs_soc * gate_soc], dim=-1)  # 加权拼接
        fused = self.tanh(self.W_fusion(combined))  # 非线性融合 [batch, dim]
        
        # 步骤3：残差连接与归一化
        output = self.layer_norm(fused + embs_soc)  # 保留社交图基础特征
        
        return output
    '''