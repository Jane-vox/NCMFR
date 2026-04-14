import numpy as np
import torch
import torch.nn.functional as F
import os 
import datetime

np.random.seed(0)


def bpr_loss(batch_user_emb, batch_pos_emb, batch_neg_emb):
    ## lightgcn
    pos_scores = torch.mul(batch_user_emb, batch_pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(batch_user_emb, batch_neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)
    
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    return loss

def reg_loss(batch_user_emb_ego, batch_pos_emb_ego, batch_neg_emb_ego):
    reg_loss = (1/2)*(batch_user_emb_ego.norm(2).pow(2) + 
                    batch_pos_emb_ego.norm(2).pow(2)  +
                    batch_neg_emb_ego.norm(2).pow(2))/float(batch_user_emb_ego.shape[0])
    return reg_loss

def generate_training_data(dataset):
    # 训练数据
    S = []
    np.random.shuffle(dataset.trainUsers)
    for user in dataset.trainUsers:
        pos_list = dataset.uu_dict[user] 
        pos_len = len(pos_list)
        if pos_len == 0:
                continue
        posindex = np.random.randint(0, pos_len) 
        pos = pos_list[posindex]
        while True:
            neg = np.random.randint(0, dataset.n_user)
            if neg in pos_list:
                continue
            else:
                break
        one_user = np.array([user, pos, neg])
        S.append(one_user)
    S = np.array(S)
    return S

def generate_cf_data(dataset, trainUsers):
    # 训练数据
    S = []
    for user in trainUsers:
        pos_list = dataset.ui_dict[user] 
        pos_len = len(pos_list)
        if pos_len == 0:
                continue
        posindex = np.random.randint(0, pos_len) 
        positem = pos_list[posindex]
        while True:
            negitem =  np.random.randint(0, dataset.m_item)
            if negitem in pos_list:
                continue
            else:
                break
        one_user = np.array([user, positem, negitem])
        S.append(one_user)
    S = np.array(S)

    return S


def get_add_dele_prob(args, users_D):
    add_prob = 1 / (1 + np.exp(users_D / args.r_add))
    dele_prob = (np.exp(users_D / args.r_dele) - np.exp(-users_D / args.r_dele)) / (np.exp(users_D / args.r_dele) + np.exp(-users_D / args.r_dele))
    return add_prob, dele_prob

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def cal_cosine_self(z, subgraph):
    indices = subgraph._indices()

    idx_0 = indices[0]
    z_0 = z[idx_0]

    idx_1 = indices[1]
    z_1 = z[idx_1]
    sim = torch.cosine_similarity(z_0, z_1, -1)
    return sim

def cal_cosine(x, y):
    x = torch.unsqueeze(x, 1)
    x = x.repeat(1, y.shape[0], 1)
    sim_matrix = torch.cosine_similarity(x, y, -1)
    return sim_matrix

def get_timestamp_folder_name():
    # 获取当前时间，并格式化为YYYY-MM-DD_HH-MM-SS的字符串
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def write_results(file_name, results):
    # 确保文件夹存在，如果不存在则创建
    folder = os.path.dirname(file_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 写入文件
    with open(file_name, "a") as f:
        f.write("\t".join(map(str, results)) + "\n")


def InfoNCE(view1, view2, temperature=0.2):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


def jaccard_index(set1, set2):
    intersection = len(set1 & set2)  # 交集的大小
    union = len(set1 | set2)  # 并集的大小
    return intersection / union if union != 0 else 0

# 计算两个用户间的jaccard指数
def calculate_jaccard(user1, user2, user_items_set):
    set1, set2 = set(user_items_set[user1]), set(user_items_set[user2])
    jaccard = jaccard_index(set1, set2)
    
    return jaccard

