import numpy as np
import torch
import torch.nn as nn
import warnings
import os
from datetime import datetime

import yaml 

from parse import set_params
from dataloader import Loader
import utils
from model import Inac_rec
from evaluation import Evaluation

np.random.seed(2025)


warnings.filterwarnings('ignore')
args = set_params()
# args.device = torch.device("cuda")
# if torch.cuda.is_available():
#     args.device = torch.device("cuda")
# else:
#     args.device = torch.device("cpu")


class TrainFlow:
    def __init__(self, args):
        self.args = args
        self.own_str = args.dataset
        # print(self.own_str)
        self.dataset = Loader(args)
        self.best_recall = -float('inf')  # 初始化最佳 recall 值
        self.best_model_state = None  # 用于保存最佳模型
        self.patience = args.patience  # 设置耐心值
        self.wait = 0  # early stop等待的轮数

        self.val_data = self.dataset.valDict
        self.test_data = self.dataset.testDict
        # self.add_prob, self.dele_prob = utils.get_add_dele_prob(args, self.dataset.users_D) # 基于ui图里的节点度

        print("-----Prepare model------")
        self.model = Inac_rec(args, self.dataset, self.dataset.n_user, self.dataset.m_item, self.dataset.train_link).to(args.device)
        self.opt_encoder = torch.optim.Adam(self.model.encoder.parameters(), lr=args.lr_encoder)#, weight_decay=args.weight_decay)
        # self.opt_gsl = torch.optim.Adam(self.model.GSL4uu.parameters(), lr=args.lr_gsl)#, weight_decay=args.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.999)

        self.eva_val = Evaluation(args, self.dataset.n_user, self.dataset.val_link, self.args.eva_neg_num, \
            "./dataset/"+self.args.dataset+"/val_neg_links.npy") # val_neg_links.npy是负样本，在函数里面进行采样
        self.eva_test = Evaluation(args, self.dataset.n_user, self.dataset.test_link, self.args.eva_neg_num, \
            "./dataset/"+self.args.dataset+"/test_neg_links.npy")

    def eva(self, data, eva, test_flag=False):
        self.model.eval()
        all_users = np.arange(self.dataset.n_user)
        all_user_emb, all_item_emb = self.model.get_emb(self.model.ui_graph, self.model.uu_graph, None, None, all_users)
        all_user_emb = all_user_emb.data.cpu().numpy()
        all_item_emb = all_item_emb.data.cpu().numpy()
        # results_10指标，ndcg, hit, recall, precision
        results_10 = eva.get_result(10, all_user_emb)  # eva对应self.eva_val或self.eva_test
        results_20 = eva.get_result(20, all_user_emb)
        # print("all users: ", results_10)
        # print("all users: ", results_20)
        # 创建字典存储结果
        results_dict = {
            'ndcg10': results_10[0],
            'hit10': results_10[1],
            'recall10': results_10[2],
            'pr10': results_10[3],
            'ndcg20': results_20[0],
            'hit20': results_20[1],
            'recall20': results_20[2],
            'pr20': results_20[3]
        }

        # 打印字典结果
        print(results_dict)
        # print('ndcg10:', results_10[0], 'hit10:', results_10[1], 'recall10:', results_10[2], 'precision10:', results_10[3])
        # print('ndcg20:', results_20[0], 'hit20:', results_20[1], 'recall20:', results_20[2], 'precision20:', results_20[3])

        stamp = args.stamp
        folder_path = f'./output/{self.own_str}/{stamp}'

        # 检查文件夹是否已存在，如果存在则不需要再创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if test_flag:
            # 生成文件路径
            file_10 = f'{folder_path}/{self.own_str}_10.txt'
            file_20 = f'{folder_path}/{self.own_str}_20.txt'

            # 写入结果
            utils.write_results(file_10, results_10)
            utils.write_results(file_20, results_20)

        return results_dict

    def train(self):
        self.model.uu_graph = self.dataset.UU_Graph
        self.model.ui_graph = self.dataset.UI_Graph
        
        print("---------Start training-----------")
        for epoch in range(args.tot_epochs):
            ## Train encoder
            print("Train encoder")
            for inner_encoder_epoch in range(args.encoder_epochs):
                # 没有分batch
                S = utils.generate_training_data(self.dataset)
                S_cf = utils.generate_cf_data(self.dataset, S[:, 0])
                batch_size = int(S.shape[0] / args.train_iters)
                temp = 0
                tot_loss = []
                print("New epoch!")

                # 通过train_iters参数决定分多少个batch进行训练
                terminal = True
                while terminal:
                    a = datetime.now()
                    self.model.train()
                    self.opt_encoder.zero_grad()

                    if args.train_iters == 1:
                        curr = S
                        curr_cf = S_cf
                        terminal = False
                    else:
                        if temp + batch_size < S.shape[0]:
                            curr = S[temp : temp + batch_size]
                            curr_cf = S_cf[temp : temp + batch_size]
                            temp += batch_size
                        else:
                            curr = S[temp : ]
                            curr_cf = S_cf[temp : ]
                            terminal = False
                    batch_user_pos_neg = curr
                    batch_cf = curr_cf

                    rec_loss = self.model.train_soc(batch_user_pos_neg=batch_user_pos_neg, ui_graph=self.model.ui_graph,
                                                uu_graph=self.model.uu_graph, user_feat=None, item_feat=None)
                    loss = rec_loss
                    # cf_loss = self.model.train_cf(batch_user_pos_neg=batch_cf, ui_graph=self.model.ui_graph,
                    #                             uu_graph=self.model.uu_graph, user_feat=None, item_feat=None)
                    # loss = rec_loss + cf_loss # 各个loss权重体现在对应函数里了
                    
                    bce_loss = self.model.train_inter_gcl(batch_user_pos_neg, self.model.ui_graph, self.model.uu_graph, None, None)
                    loss = rec_loss + args.cl_lam*bce_loss
                    # loss = rec_loss + args.cl_lam*gcl_loss

                    # recon_A, recon_S, mu, logvar, s_mu, s_logvar, u_z, s_z = self.model.train_encoder2(batch_user_pos_neg, self.model.ui_graph, self.model.uu_graph)

                    # cf_loss = self.model.recon_loss(recon_A, batch_user_pos_neg[:,0], mu, logvar)
                    # soc_loss = self.model.recon_loss(recon_S, batch_user_pos_neg[:,0], s_mu, s_logvar)
                    # loss = cf_loss + soc_loss
                    
                    loss.backward()
                    self.opt_encoder.step()
                    tot_loss.append(loss.cpu().data.numpy())

                b = datetime.now() 
                print("tot_epochs ", epoch, "\tencoder_epochs ", inner_encoder_epoch, "\trec Loss ", np.array(tot_loss).mean(), "\tseconds ", (b-a).seconds)
            
            print("Get user item emb")
            all_users = np.arange(self.dataset.n_user)
            user_emb, item_emb, user_emb_ego = self.model.get_emb(self.model.ui_graph, self.model.uu_graph, None, None, all_users, temp_flag=True)
            ## Train gsl
            terminal2 = False  # 可用于消融实验修改
            while terminal2:
                print("Train gsl")
                for inner_gsl in range(args.gsl_epochs):
                    S = utils.generate_training_data(self.dataset)
                    tot_loss = []
                    print("New epoch!")
                    a = datetime.now()
                    self.model.train()
                    self.opt_gsl.zero_grad()

                    gsl_loss = self.model.train_gsl(batch_user_pos_neg=S, user_emb_ego=user_emb_ego, user_emb=user_emb, uu_dict=self.dataset.uu_dict, add_prob=self.add_prob, dele_prob=self.dele_prob)
                    gsl_loss.backward()
                    self.opt_gsl.step()
                    tot_loss.append(gsl_loss.cpu().data.numpy())
                    b=datetime.now()
                    print("tot_epochs ", epoch, "\tgsl_epochs ", inner_gsl, "\tgsl Loss ", np.array(tot_loss).mean(), "\tseconds ", (b-a).seconds)
                
        
                ## Get whole structure
                with torch.no_grad():
                    print("Get whole structure")
                    all_users = torch.arange(self.dataset.n_user)
                    final_dele_indices, final_dele_sim = self.model.re_struct(all_users, self.dataset.uu_dict, user_emb, self.dele_prob)
                    
                    self.model.uu_graph = self.model.get_whole_stru(all_users, final_dele_indices, final_dele_sim, self.dataset.n_user, None, None)
                    print("Get whole structure finish")
              
                terminal = False

            print("Evaluation")
            metrics = self.eva(self.val_data, self.eva_val, test_flag=True)
            print(metrics)
            recall10 = metrics['recall10']
            # torch.save(self.model.state_dict(), self.own_str+'.pkl')
            # 检查是否改进
            if recall10 > self.best_recall:
                print(f"New best recall@10: {recall10}")
                self.best_recall = recall10
                self.best_model_state = self.model.state_dict()  # 保存最佳模型
                self.wait = 0  # 重置等待计数
            else:
                self.wait += 1
                print(f"No improvement. Wait: {self.wait}/{self.patience}")
            
            # Early Stopping 检查
            if self.wait >= self.patience:
                print("Early stopping triggered.")
                break

        # 保存最佳模型
        print("Saving best model...")
        if self.best_model_state is not None:
            torch.save(self.best_model_state, f'./output/{self.own_str}/{self.args.stamp}/best_model.pkl')
    
        ## test ##
        print("Testing best model...")
        self.model.load_state_dict(self.best_model_state)
        metrics = self.eva(self.test_data, self.eva_test, test_flag=True)
        print(metrics)

        # 保存训练参数
        args_dict = vars(self.args)
        with open(f'./output/{self.own_str}/{self.args.stamp}/args.yaml', 'w') as file:
            yaml.dump(args_dict, file, default_flow_style=False)
        print(f'参数已保存.')            


if __name__ == '__main__':
    train = TrainFlow(args)
    train.train()
