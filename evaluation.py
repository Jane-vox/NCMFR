import numpy as np 
from collections import defaultdict
import torch
import torch.nn as nn
import math
import os
np.random.seed(0)


class Evaluation:
    def __init__(self, args, user_num, test_link, eva_neg_num, neg_links):
        self.user_num = user_num
        self.args = args

        self.pos_links = test_link[np.argsort(test_link[:,0])] #排序
        self.pos_links_len = defaultdict(int)
        self.neg_links = []
        self.test_uu_pos_dict = defaultdict(list)

        # 无向图处理
        for one in self.pos_links:
            self.test_uu_pos_dict[one[0]].append(one[1])
            self.pos_links_len[one[0]]+=1

            self.test_uu_pos_dict[one[1]].append(one[0])
            self.pos_links_len[one[1]]+=1

        if os.path.exists(neg_links):
            self.neg_links = np.load(neg_links)
        else:
            # 随机采样负样本
            for k, v in self.test_uu_pos_dict.items():
                for _ in range(eva_neg_num):
                    j = np.random.randint(self.user_num)
                    while j in v:
                        j = np.random.randint(self.user_num)
                    self.neg_links.append([k, j])
            self.neg_links = np.array(self.neg_links)      
        # self.index_inact = index_inact
        
    def getIdcg(self, length):
        idcg = 0.0
        for i in range(length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        return idcg

    def getDcg(self, value):
        dcg = math.log(2) / math.log(value + 2)
        return dcg

    def getHr(self, value):
        hit = 1.0
        return hit

    # 评价指标计算
    def get_result(self, at_value, user_embed):
        self.at_value = at_value  # top-k
        pos_sim = (user_embed[self.pos_links[:, 0]] * user_embed[self.pos_links[:, 1]]).sum(-1)
        neg_sim = self.batch_neg_sim(user_embed=user_embed, neg_links=self.neg_links, batch_size=self.args.valbatch) 
        user_sim = defaultdict(list)

        test_user_pos = self.pos_links[:, 0]
        for i in range(len(test_user_pos)):
            user_sim[test_user_pos[i]].append(pos_sim[i])

        test_user_neg = self.neg_links[:, 0]
        for i in range(len(test_user_neg)):
            user_sim[test_user_neg[i]].append(neg_sim[i])
        
        ndcg = []
        recall = []
        precision = []
        hr = 0
        hr_len = 0
        # ndcg_inac = []
        # recall_inac = []
        # precision_inac = []
        # hr_inac = 0
        # hr_inac_len = 0

        for user, sims in user_sim.items():
            sort_index = np.argsort(sims)
            sort_index = sort_index[::-1]

            user_ndcg_list = []
            hits_num = 0
            for idx in range(self.at_value):
                ranking = sort_index[idx]
                if ranking < self.pos_links_len[user]:
                    hits_num += 1
                    user_ndcg_list.append(self.getDcg(idx))
            hr += hits_num
            hr_len += self.pos_links_len[user]
            recall.append(hits_num / self.pos_links_len[user])
            precision.append(hits_num / self.at_value)
            
            target_length = min(self.pos_links_len[user], self.at_value)
            idcg = self.getIdcg(target_length)
            ndcg.append(np.sum(user_ndcg_list) / idcg)
            
            # if user in self.index_inact:
            #     hr_inac += hits_num
            #     hr_inac_len += self.pos_links_len[user]
            #     recall_inac.append(hits_num / self.pos_links_len[user])
            #     precision_inac.append(hits_num / self.at_value)
            #     ndcg_inac.append(np.sum(user_ndcg_list) / idcg)
        recall = np.array(recall)
        precision = np.array(precision)
        ndcg = np.array(ndcg)
        # recall_inac = np.array(recall_inac)
        # precision_inac = np.array(precision_inac)
        # ndcg_inac = np.array(ndcg_inac)
        return (ndcg.mean(), hr/(hr_len+1e-8), recall.mean(), precision.mean())
    
    def batch_neg_sim(self, user_embed, neg_links, batch_size):
        neg_sim = []
        num_samples = neg_links.shape[0]

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_neg = neg_links[start:end]

            neg_sim.append((user_embed[batch_neg[:, 0]] * user_embed[batch_neg[:, 1]]).sum(-1))

        return np.concatenate(neg_sim)

