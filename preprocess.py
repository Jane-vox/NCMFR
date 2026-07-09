import numpy as np
import pandas as pd
import os
import yaml

def process_dataset(dataset_name, raw_dir, out_dir, seed=2025):
    print(f"\n=========================================")
    print(f"正在处理 {dataset_name} 数据集 (Seed={seed})...")
    
    if not os.path.exists(raw_dir):
        print(f"找不到原始数据文件夹: {raw_dir}")
        return
        
    os.makedirs(out_dir, exist_ok=True)

    if dataset_name == 'douban-book':
        inter_filename = "user_book.dat"  # librahu 仓库的 U-I 命名
        net_filename = "user_user.dat"    # librahu 仓库的 U-U 命名
    else:
        inter_filename = f"{dataset_name}.inter" # Sherry-XLL 仓库的 U-I 命名
        net_filename = f"{dataset_name}.net"     # Sherry-XLL 仓库的 U-U 命名

    # 2. 读取并处理 U-I 交互图
    inter_path = os.path.join(raw_dir, inter_filename)
    try:
        inter_df = pd.read_csv(inter_path, sep=r'\s+', header=None, comment='#')
        if isinstance(inter_df.iloc[0, 0], str):
            inter_df = inter_df.iloc[1:]

        inter_data = inter_df.iloc[:, :2].values.astype(int)
        np.save(os.path.join(out_dir, 'inter.npy'), inter_data)
        print(f"成功提取 U-I 交互图，总交互数: {len(inter_data)}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {inter_path}")
        return

    # 3. 读取并处理 U-U 社交图
    net_path = os.path.join(raw_dir, net_filename)
    try:
        net_df = pd.read_csv(net_path, sep=r'\s+', header=None, comment='#')
        if isinstance(net_df.iloc[0, 0], str):
            net_df = net_df.iloc[1:]

        net_data = net_df.iloc[:, :2].values.astype(int)
        print(f"成功提取 U-U 社交图，总关系数: {len(net_data)}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {net_path}")
        return

    # 4. 按 8:1:1 划分
    np.random.seed(seed)
    np.random.shuffle(net_data)

    total_edges = len(net_data)
    train_end = int(total_edges * 0.8)
    val_end = int(total_edges * 0.9)

    train_link = net_data[:train_end]
    val_link = net_data[train_end:val_end]
    test_link = net_data[val_end:]

    np.save(os.path.join(out_dir, 'train.npy'), train_link)
    np.save(os.path.join(out_dir, 'val.npy'), val_link)
    np.save(os.path.join(out_dir, 'test.npy'), test_link)

    # 5. 统计用户数和物品数并生成 info.yaml
    max_user_inter = inter_data[:, 0].max()
    max_user_net = max(net_data[:, 0].max(), net_data[:, 1].max())
    n_user = int(max(max_user_inter, max_user_net) + 1)
    
    # 获取全局最大的 Item ID
    m_item = int(inter_data[:, 1].max() + 1)

    info = {'user': n_user, 'item': m_item}
    with open(os.path.join(out_dir, 'info.yaml'), 'w') as f:
        yaml.dump(info, f)



if __name__ == '__main__':
    raw_base_dir = "./data"
    output_base_dir = "./dataset"

    global_seed = 2025

    # 依次处理三个数据集
    process_dataset("yelp1", f"{raw_base_dir}/yelp1", f"{output_base_dir}/yelp1", seed=global_seed)
    process_dataset("flickr", f"{raw_base_dir}/flickr", f"{output_base_dir}/flickr", seed=global_seed)
    process_dataset("douban-book", f"{raw_base_dir}/douban-book", f"{output_base_dir}/douban-book", seed=global_seed)
    
    print("\n所有数据处理完毕！")