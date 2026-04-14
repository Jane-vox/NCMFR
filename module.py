import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import os

from layer import MLP, Propagation
from utils import *


class inter_denoise_gate(nn.Module):
    def __init__(self, args):
        super(inter_denoise_gate, self).__init__()
        self.W_cf = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.W_soc = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.W_mix = nn.Linear(args.hidden_dim, args.hidden_dim)
        
        self.W_en = nn.Linear(int(2*args.hidden_dim), args.hidden_dim)
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, embs_cf, embs_soc):
        embs = self.W_cf(embs_cf)+self.W_soc(embs_soc)+self.W_mix(embs_soc*embs_cf)
        forget = self.sig(embs)  
        
        embs = self.W_en(torch.cat((embs_cf, embs_soc), -1))
        enhance = self.sig(embs)
        out = forget * embs_soc + enhance*self.tanh(self.W_soc(embs_soc))
        return out
    


class MultiHeadAttention(nn.Module):
    def __init__(self, args, input_dim, num_heads, num_users, dropout_rate, ui_dict, train_link):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.output_linear = nn.Linear(input_dim, input_dim)
        self.num_users = num_users
        self.args = args
        self.ui_dict = ui_dict
        self.train_link = train_link
        self.attention_scores = self.getAttention("./dataset/"+args.dataset+"/attention_scores.pth")
    
    def getAttention(self, attention_scores_path):
        users = range(self.num_users)
        # Softmax = nn.Softmax(dim=-1)

        if os.path.exists(attention_scores_path):
            attention_scores = torch.load(attention_scores_path)
        else:
            attention_scores = torch.zeros((len(users), len(users)))
            for row in self.train_link:
                # 计算每对用户之间的 Jaccard 相似度
                user1, user2 = row
                jaccard_similarity = calculate_jaccard(user1, user2, self.ui_dict)
                attention_scores[user1, user2] = jaccard_similarity
                attention_scores[user2, user1] = jaccard_similarity
        
            torch.save(attention_scores, attention_scores_path)
        
        return attention_scores.cuda()

    def forward(self, social_emb, preference_emb, uu_graph, train=True):

        final_embeddings = torch.zeros_like(social_emb)
        for i in range(self.num_users):
            # weighted_social_sum = torch.sum(self.attention_scores[i].unsqueeze(1) * social_emb, dim=0)
            weighted_preference_sum = torch.sum(self.attention_scores[i].unsqueeze(1) * preference_emb, dim=0)
            final_embeddings[i] = weighted_preference_sum

        # final_embeddings = self.output_linear(final_embeddings)
        # final_embeddings = self.user_emb_map(torch.cat([social_emb, final_embeddings], -1))
        final_embeddings = self.prop_uu(uu_graph, final_embeddings, train) # uu

        return self.dropout(final_embeddings)
