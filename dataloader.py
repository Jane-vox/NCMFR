import os
from os.path import join
import sys
import torch
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from collections import defaultdict
from utils import _convert_sp_mat_to_sp_tensor


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, args):
        # train or test
        self.args = args
        path="./dataset/"+args.dataset
        print(f'loading [{path}]')
        self.path = path

        # self.cluster_map = np.load(path + '/cluster_map.npy')  # active user的聚类类别
        # self.index_inact = np.load(path + '/index_inact.npy')  # inact索引
        # self.index_act = np.load(path + '/index_act.npy')
        # 将u-u图划分为训练，验证，测试集
        self.train_link = np.load(path + '/train.npy')  
        # print(self.train_link)
        self.val_link = np.load(path + '/val.npy')
        self.test_link = np.load(path + '/test.npy')
        self.inter_link = np.load(path + '/inter.npy')  # u-i交互图
        with open(path+'/info.yaml', 'r') as file:
            info = yaml.load(file, Loader=yaml.FullLoader)
        self.n_user = info['user']
        self.m_item = info['item']

        self.trainUsers = np.unique(np.concatenate((self.train_link[:, 0], self.train_link[:, 1])))
        self.valUsers = np.unique(np.concatenate((self.val_link[:, 0], self.val_link[:, 1])))
        self.testUsers = np.unique(np.concatenate((self.test_link[:, 0], self.test_link[:, 1])))
        
        # uu_dict(当做无向图处理)
        self.uu_dict = defaultdict(list)
        for i in self.train_link:
            self.uu_dict[i[0]].append(i[1])
            self.uu_dict[i[1]].append(i[0])    

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.inter_link[:, 0])), (self.inter_link[:, 0], self.inter_link[:, 1])),
                                      shape=(self.n_user, self.m_item))
        self.UI_Graph = self.getSparseGraph_UI()
        self.UU_Graph = self.getSparseGraph_UU(self.train_link)
        
        # 用户和物品的度,ui图
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze() 
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        self.ui_dict = self.getUserPosItems(range(self.n_user))

        self.valDict = self.__build_val()  # dict: {user: [users]}
        self.testDict = self.__build_test()
        print(f"{args.dataset} is ready to go")
        
    def getSparseGraph_UI(self):
        print("loading UI matrix")
        try:
            pre_adj_mat = sp.load_npz(self.path + '/adj_ui.npz')
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except :
            print("generating UI matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
            adj_mat = adj_mat.tolil() # shape=(55579,55579)
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end-s}s, saved norm_mat...")
            sp.save_npz(self.path + '/adj_ui.npz', norm_adj)

        Graph = _convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().cuda()
        return Graph

    def getSparseGraph_UU(self, social_link):
        
        print("loading UU matrix")
        # 处理无向图：添加逆边
        reverse_edges = social_link[:, [1, 0]]  # 交换每对边的起点和终点
        social_link = np.vstack([social_link, reverse_edges])  # 合并原始边和逆向边

        try:
            pre_adj_mat = sp.load_npz(self.path + '/adj_uu.npz')
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except :
            print("generating UU matrix")
            s = time()
            adj_mat = sp.coo_matrix((np.ones(social_link.shape[0]), (social_link[:, 0], social_link[:, 1])), shape=(self.n_user, self.n_user))
            adj_mat = adj_mat.todok()
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end-s}s, saved norm_mat...")
            sp.save_npz(self.path + '/adj_uu.npz', norm_adj)

        Graph = _convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().cuda()
        return Graph

    def __build_val(self):
        """
        return:
            dict: {user: [users]}
        """
        val_data = defaultdict(list)
        for i in self.val_link:
            val_data[i[0]].append(i[1])
            val_data[i[1]].append(i[0])    
        # print("val_data: ", val_data)
        return val_data

    def __build_test(self):
        """
        return:
            dict: {user: [users]}
        """
        test_data = defaultdict(list)
        for i in self.test_link:
            test_data[i[0]].append(i[1])
            test_data[i[1]].append(i[0])    
        return test_data

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems