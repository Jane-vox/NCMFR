import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from layer import MLP, Propagation
from utils import bpr_loss, reg_loss, InfoNCE
from gsl_uu import GSL4uu
from utils import _convert_sp_mat_to_sp_tensor
import scipy.sparse as sp

from encoder import *
from module import *

class Inac_rec(nn.Module):
    def __init__(self, args, dataset, num_users, num_items, train_link):
        super(Inac_rec, self).__init__()
        self.args = args
        # self.GSL4uu = GSL4uu(args.hidden_dim, args.m_head, args.dropout, args.min_keep, args.max_add, \
        #     args.pseudo_num, args.pseudo_lam, args.tau, args.edge_emb_flag)

        self.encoder = Encoder(args, num_users, num_items) 
        # self.encoder2 = Encoder2(args, num_users, num_items) 
        self.num_users = num_users
        self.num_items = num_items

        self.weight = args.add_weight
        self.ui_dict = dataset.ui_dict
        self.uu_dict = dataset.uu_dict
        # self.multihead = MultiHeadAttention(args=args, input_dim=args.hidden_dim, num_heads=2, num_users=num_users, dropout_rate=0.0, ui_dict=ui_dict, train_link=train_link)
        self.gate = inter_denoise_gate(args)
        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
       

    
    def train_soc(self, batch_user_pos_neg, ui_graph, uu_graph, user_feat=None, item_feat=None):
        
        batch_user, batch_pos, batch_neg = batch_user_pos_neg[:, 0], batch_user_pos_neg[:, 1], batch_user_pos_neg[:, 2]

        user_final_emb, user_ui_emb, item_emb, user_emb_ego, item_emb_ego = self.encoder(ui_graph, uu_graph, user_feat, item_feat)
        rec_loss = bpr_loss(user_final_emb[batch_user], user_final_emb[batch_pos], user_final_emb[batch_neg])

        reg = reg_loss(user_emb_ego[batch_user], user_emb_ego[batch_pos], user_emb_ego[batch_neg])

        return rec_loss*self.args.soc_lam + reg*self.args.weight_decay
    
    
    def train_cf(self, batch_user_pos_neg, ui_graph, uu_graph, user_feat=None, item_feat=None):

        batch_user, batch_pos, batch_neg = batch_user_pos_neg[:, 0], batch_user_pos_neg[:, 1], batch_user_pos_neg[:, 2]
        
        user_final_emb, user_ui_emb, item_emb, user_emb_ego, item_emb_ego = self.encoder(ui_graph, uu_graph, user_feat, item_feat)
        cf_loss = bpr_loss(user_final_emb[batch_user], item_emb[batch_pos], item_emb[batch_neg])

        reg = reg_loss(user_emb_ego[batch_user], item_emb_ego[batch_pos], item_emb_ego[batch_neg])

        return cf_loss*self.args.cf_lam + reg*self.args.weight_decay
    
    def get_emb(self, ui_graph, uu_graph, user_feat, item_feat, users, temp_flag=False):
        # 模型验证获取最终embedding
        if temp_flag:
            user_final_emb, item_emb, user_emb_ego = self.encoder(ui_graph, uu_graph, user_feat, item_feat, train=False, temp_flag=temp_flag)
            return user_final_emb.detach()[users], item_emb.detach(), user_emb_ego.detach()
        else:
            user_final_emb, item_emb = self.encoder(ui_graph, uu_graph, user_feat, item_feat, train=False, temp_flag=temp_flag)
            return user_final_emb.detach()[users], item_emb.detach()
    

    def get_whole_stru(self, unique_user, final_dele_indices, final_dele_sim, user_num, final_add_indices=None, final_add_sim=None):
        # 更新的social图
        dele_graph = torch.sparse_coo_tensor(final_dele_indices.t(), final_dele_sim, (user_num, user_num)).cuda()
        dele_graph = torch.sparse.softmax(dele_graph, 1)
        # add_graph = torch.sparse_coo_tensor(final_add_indices.t(), final_add_sim, (user_num, user_num)).cuda()
        # add_graph = torch.sparse.softmax(add_graph, 1)
        self_loop = torch.sparse_coo_tensor(torch.cat([unique_user.unsqueeze(0), unique_user.unsqueeze(0)]), torch.ones_like(unique_user), (self.num_users, self.num_users)).cuda()
        # batch_graph = 1/2 * self_loop + 1/2 * (self.weight*add_graph + (1-self.weight)*dele_graph)
        # batch_graph = 1/2 * self_loop + 1/2 * (1-self.weight)*dele_graph
        batch_graph = dele_graph
        return batch_graph

    def train_graph_generator(self, user_emb_ego, batch_user_pos_neg, batch_act_user, batch_inact_user, uu_dict, user_emb, item_emb, add_prob, dele_prob):
        batch_user, batch_pos, batch_neg = batch_user_pos_neg[:, 0], batch_user_pos_neg[:, 1], batch_user_pos_neg[:, 2]
        # 获取加边和删边的索引
        final_dele_indices, final_dele_sim, final_add_indices, final_add_sim = self.get_stru(user_emb_ego, batch_user, user_emb, uu_dict, add_prob, dele_prob)
        
        unique_user = torch.LongTensor(np.sort(list(set(batch_user))))
        batch_graph = self.get_whole_stru(unique_user, final_dele_indices, final_dele_sim, final_add_indices, final_add_sim, self.num_users)
        uu_emb = torch.sparse.mm(batch_graph, user_emb)
        user_final_emb = self.encoder.user_emb_map(torch.cat([user_emb_ego[batch_user], user_emb[batch_user], uu_emb[batch_user]], -1)) # user自身嵌入，UI图user嵌入，social图更新后user嵌入       
        rec_loss = bpr_loss(user_final_emb, item_emb[batch_pos], item_emb[batch_neg])

        mimic_loss = self.GSL4uu.mimic_learning(batch_act_user, batch_inact_user, user_emb, self.cluster_map, self.cluster_index)
        return rec_loss + self.args.loss_lam * mimic_loss
    
    def get_stru(self, user_emb_ego, batch_user, user_emb_ori, uu_dict, add_prob, dele_prob, edge_emb=None):
        # social图增删边实现函数
        target = np.sort(list(set(batch_user)))

        # target user的邻居
        link_target_nei = []
        target_nei = []
        for u in target:
            u_nei = uu_dict[u]
            target_nei += u_nei
            t = [u]*len(u_nei)
            link_target_nei.append(np.array([t, u_nei]).T)
            link_target_nei.append(np.array([u_nei, t]).T)
        link_target_nei = np.vstack(link_target_nei)
        target_nei = np.sort(list(set(target_nei)))

        # target user的邻居的邻居   
        link_nei_nei = []
        target_nei_nei = []
        for u in target_nei:
            u_nei = uu_dict[u]
            target_nei_nei += u_nei
            t = [u]*len(u_nei)
            link_nei_nei.append(np.array([t, u_nei]).T)
        link_nei_nei = np.vstack(link_nei_nei)
        target_nei_nei = np.sort(list(set(target_nei_nei)))

        cluster = self.cluster_index
        link_cluster_nei = []
        cluster_nei = []
        cluster_final_map = {}
        for i in range(len(cluster)):
            u_nei = uu_dict[cluster[i]]
            cluster_nei += u_nei
            t = [cluster[i]]*len(u_nei)
            link_cluster_nei.append(np.array([t, u_nei]).T)
            cluster_final_map[i] = cluster[i]
        link_cluster_nei = np.vstack(link_cluster_nei)
        cluster_nei = np.sort(list(set(cluster_nei)))

        all_nodes = np.sort(list(set(list(target) + list(target_nei) + list(target_nei_nei) + list(cluster) + list(cluster_nei))))
        all_nodes_num = len(all_nodes)
        all_nodes_map = {}
        for n in range(len(all_nodes)):
            all_nodes_map[all_nodes[n]] = n
        all_links_ = np.vstack([link_target_nei, link_nei_nei, link_cluster_nei])
        all_links = []  # 当前子图的去重边集合，重新映射到 all_nodes_map 的索引空间后得到。
        for one_link in all_links_:
            all_links.append((all_nodes_map[one_link[0]], all_nodes_map[one_link[1]]))
        all_links = np.array(list(set(all_links)))
        all_links = all_links[np.argsort(all_links[:, 0])].T
        batch_subgraph = torch.sparse_coo_tensor(all_links, [1]*all_links.shape[1], (all_nodes_num, all_nodes_num)).cuda() # 大的采样图

        tn_nodes = np.sort(list(set(list(target) + list(target_nei))))
        tn_nodes_ = np.array([all_nodes_map[n] for n in tn_nodes])
        tn_nodes_num = len(tn_nodes)
        tn_nodes_map = {} # 局部子图中每个节点到索引的映射。
        tn_final_map = {} # 局部子图中索引到原始节点 ID 的反向映射。
        for n in range(len(tn_nodes_)):
            tn_nodes_map[tn_nodes_[n]] = n
            tn_final_map[n] = tn_nodes[n]
        tn_links = []
        for one_link in link_target_nei:
            tn_links.append((tn_nodes_map[all_nodes_map[one_link[0]]], tn_nodes_map[all_nodes_map[one_link[1]]]))
        tn_links = np.array(list(set(tn_links)))
        tn_links = tn_links[np.argsort(tn_links[:, 0])].T
        tn_subgraph = torch.sparse_coo_tensor(tn_links, [1]*tn_links.shape[1], (tn_nodes_num, tn_nodes_num)).cuda()  # 小的采样图
        
        ## T=0 ##
        # 第一轮的增删边涵盖了更广泛的节点集合，为了捕捉更多的结构信息和潜在的边。
        prob_dele_edge = dele_prob[all_nodes]
        prob_add_edge = add_prob[all_nodes]
        dele_final_indices, dele_final_sim, add_final_indices, add_final_sim = self.GSL4uu.new_stru(user_emb_ori, prob_dele_edge, prob_add_edge, all_nodes, cluster, batch_subgraph) 
        # 增删边对稀疏矩阵进行更新，并归一化       
        dele_sim_t0 = torch.sparse_coo_tensor(dele_final_indices, dele_final_sim[0], (all_nodes_num, all_nodes_num))
        dele_sim_t0 = torch.sparse.softmax(dele_sim_t0, 1)
        add_sim_t0 = torch.sparse_coo_tensor(add_final_indices, add_final_sim[0], (all_nodes_num, len(cluster)))
        add_sim_t0 = torch.sparse.softmax(add_sim_t0, 1)
        
        # 嵌入更新
        ori_feat = user_emb_ori[all_nodes]
        cluster_ori_feat = user_emb_ori[cluster]
        dele_emb = torch.sparse.mm(dele_sim_t0, ori_feat)
        add_emb = torch.sparse.mm(add_sim_t0, cluster_ori_feat)
        user_emb_t0 = 1/2 * ori_feat +  1/2 * (self.weight * add_emb + (1-self.weight) * dele_emb)
        user_emb_t0 = self.encoder.user_emb_map(torch.cat([user_emb_ego[all_nodes], ori_feat, user_emb_t0], -1)) 

        ## T=1 ##
        # 第二轮的增删边则是在第一轮结果的基础上缩小了范围，专注于目标节点及其直接邻居，这样做可能是为了在局部范围内进一步优化结构。
        prob_dele_edge = dele_prob[tn_nodes]
        prob_add_edge = add_prob[tn_nodes]
        # user_emb_ori_ = user_emb_ori[all_nodes]
        cluster_ = np.array([all_nodes_map[c] for c in cluster])
        dele_final_indices, dele_final_sim, add_final_indices, add_final_sim = self.GSL4uu.new_stru(user_emb_t0, prob_dele_edge, prob_add_edge, tn_nodes_, cluster_, tn_subgraph)
        dele_final_indices = dele_final_indices.data.cpu().numpy()
        add_final_indices = add_final_indices.data.cpu().numpy()

        dele_sele = []
        final_dele_indices = []
        for i in range(dele_final_indices.shape[1]):
            if tn_final_map[dele_final_indices[0, i]] in target:
                dele_sele.append(i)
                final_dele_indices.append([tn_final_map[dele_final_indices[0, i]], tn_final_map[dele_final_indices[1, i]]])
        final_dele_indices = torch.LongTensor(final_dele_indices).cuda()
        final_dele_sim = dele_final_sim[0, dele_sele]
        
        add_sele = []
        final_add_indices = []
        for i in range(add_final_indices.shape[1]):
            if tn_final_map[add_final_indices[0, i]] in target:
                add_sele.append(i)
                final_add_indices.append([tn_final_map[add_final_indices[0, i]], cluster_final_map[add_final_indices[1, i]]])
        final_add_indices = torch.LongTensor(final_add_indices).cuda()
        final_add_sim = add_final_sim[0, add_sele]
        return final_dele_indices, final_dele_sim, final_add_indices, final_add_sim
   
    def train_gsl(self, batch_user_pos_neg, user_emb_ego, user_emb, uu_dict, add_prob, dele_prob):
        # social图删边，去除脏数据
        batch_user, batch_pos, batch_neg = batch_user_pos_neg[:, 0], batch_user_pos_neg[:, 1], batch_user_pos_neg[:, 2]
        
        final_dele_indices, final_dele_sim = self.re_struct(batch_user, uu_dict, user_emb, dele_prob)
        
        unique_user = torch.LongTensor(np.sort(list(set(batch_user))))
        batch_graph = self.get_whole_stru(unique_user, final_dele_indices, final_dele_sim, self.num_users, None, None)
        uu_emb = torch.sparse.mm(batch_graph, user_emb)
        # user_final_emb = self.encoder.user_emb_map(torch.cat([user_emb_ego[batch_user], user_emb[batch_user], uu_emb[batch_user]], -1)) # user自身嵌入，UI图user嵌入，social图更新后user嵌入
        user_final_emb = uu_emb[batch_user]       
        gsl_loss = bpr_loss(user_final_emb, user_emb[batch_pos], user_emb[batch_neg])
        # cl_loss = InfoNCE(user_emb[batch_user], user_final_emb, temperature=0.2)
        # gsl_loss += cl_loss

        return gsl_loss

    def re_struct(self, batch_user, uu_dict, user_emb_ori, dele_prob):
        # social图增删边实现函数
        target = np.sort(list(set(batch_user)))
        # target user的邻居target_nei，包含的边link_target_nei
        link_target_nei = []
        target_nei = []
        for u in target:
            u_nei = uu_dict[u]
            target_nei += u_nei
            t = [u]*len(u_nei)
            link_target_nei.append(np.array([t, u_nei]).T)
            link_target_nei.append(np.array([u_nei, t]).T)
        link_target_nei = np.vstack(link_target_nei)
        target_nei = np.sort(list(set(target_nei))) #从小到大排序

         # 局部子图节点
        tn_nodes = np.sort(list(set(list(target) + list(target_nei))))
        tn_nodes_ = np.array(tn_nodes)  # 保留原始节点 ID
        tn_nodes_num = len(tn_nodes)

        # 节点映射
        tn_nodes_map = {n: i for i, n in enumerate(tn_nodes)} # 节点映射{node:index}
        tn_final_map = {i: n for i, n in enumerate(tn_nodes)} # 反向映射获取节点id：{index:node}

        tn_links = []
        for one_link in link_target_nei:
            tn_links.append((tn_nodes_map[one_link[0]], tn_nodes_map[one_link[1]]))
        tn_links = np.array(list(set(tn_links)))
        tn_links = tn_links[np.argsort(tn_links[:, 0])].T
        tn_subgraph = torch.sparse_coo_tensor(tn_links, [1]*tn_links.shape[1], (tn_nodes_num, tn_nodes_num)).cuda()

        prob_dele_edge = dele_prob[tn_nodes]
        
        # dele_final_indices表示保留的边，dele_final_sim表示保留边的相似度
        dele_final_indices, dele_final_sim = self.GSL4uu.new_stru(user_emb_ori, prob_dele_edge, None, tn_nodes_, None, tn_subgraph)
        dele_final_indices = dele_final_indices.data.cpu().numpy()

        dele_sele = []
        final_dele_indices = []
        for i in range(dele_final_indices.shape[1]):
            if tn_final_map[dele_final_indices[0, i]] in target:
                dele_sele.append(i)
                final_dele_indices.append([tn_final_map[dele_final_indices[0, i]], tn_final_map[dele_final_indices[1, i]]])
        final_dele_indices = torch.LongTensor(final_dele_indices).cuda()
        final_dele_sim = dele_final_sim[0, dele_sele]

        return final_dele_indices, final_dele_sim
    

    def train_inter_gcl(self, batch_user_pos_neg, ui_graph, uu_graph, user_feat=None, item_feat=None): 
        all_layer_1, all_layer_2, _, user_emb_ego, _ = self.encoder(ui_graph, uu_graph, user_feat, item_feat)

        all_layer_1 = F.normalize(all_layer_1, dim=1) #u-u图里user嵌入
        all_layer_2 = F.normalize(all_layer_2, dim=1) #u-i图里user嵌入

        batch_user, batch_pos, batch_neg = batch_user_pos_neg[:, 0], batch_user_pos_neg[:, 1], batch_user_pos_neg[:, 2]

        user_emb_1 = all_layer_1[batch_user]  # uu
        user_emb_2 = all_layer_2[batch_user]  # ui

#       inter-domain denoising
        user_emb = self.gate(user_emb_2, user_emb_1)  # 融合cf信息的新user嵌入

        similarity_matrix = torch.matmul(user_emb, user_emb.t())
        predicted_probs = self.sigmoid(similarity_matrix)  # predict

        labels = torch.zeros((self.num_users, self.num_users))
        for i, neighbors in self.uu_dict.items():
            for j in neighbors:
                labels[i, j] = 1
        labels = labels[batch_user][:, batch_user]
        bce_loss = self.criterion(predicted_probs, labels.to(self.args.device))
        reg = reg_loss(user_emb_ego[batch_user], user_emb_ego[batch_pos], user_emb_ego[batch_neg])

        # rec_loss = bpr_loss(user_emb, all_layer_1[batch_pos], all_layer_1[batch_neg])
        # infonce_loss = self._cl_loss(user_emb, F.normalize(all_layer_1[batch_user], dim=1))  # 拉近 融合cf信息的新user嵌入 和 原始uu图上的user嵌入

        return bce_loss + reg*self.args.weight_decay
    
    # def _cl_loss(self, emb_1, emb_2):       
    #     pos = torch.sum(emb_1*emb_2, dim=-1) 
    #     tot = torch.matmul(emb_1, torch.transpose(emb_2, 0, 1)) 
    #     gcl_logits = tot - pos[:, None]                            
    #     #InfoNCE Loss
    #     clogits = torch.logsumexp(gcl_logits / self.args.gcl_temp, dim=1)
    #     infonce_loss = torch.mean(clogits)
    #     return infonce_loss
   
    '''
    # def train_encoder2(self, batch_user_pos_neg, ui_graph, uu_graph):
    #     inputs, batch_pos, batch_neg = batch_user_pos_neg[:, 0], batch_user_pos_neg[:, 1], batch_user_pos_neg[:, 2]

    #     mu, logvar = self.encoder2.encode(ui_graph)   # ui图    
    #     s_mu, s_logvar = self.encoder2.social_encode(uu_graph)  # uu图
    #     u_z = self.encoder2.reparameterize(mu[inputs], logvar[inputs]) 
    #     s_z = self.encoder2.reparameterize(s_mu[inputs], s_logvar[inputs])
        
    #     all_z =torch.cat([s_z, u_z], dim=1)
    #     score = self.encoder2.attention_dense(all_z)
    #     z = score * s_z + (1 - score) * u_z

    #     recon_A = self.encoder2.decode(z)
    #     recon_S = self.encoder2.social_decode(s_z)
        
    #     return recon_A, recon_S, mu, logvar, s_mu, s_logvar, u_z, s_z
            
    # def recon_loss(self, recon_x, x, mu, logvar, anneal=1.0):
    #     x = torch.FloatTensor(x).to(self.args.device)
    #     BCE = - torch.mean(torch.sum(F.log_softmax(recon_x[:, :self.num_users], 1) * x, -1))  # multi
    #     KLD = - 0.5 / recon_x.size(0) * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    #     return BCE + anneal * KLD
    '''       
    
