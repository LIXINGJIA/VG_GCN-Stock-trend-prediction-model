import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
#from data_provider.uea import subsample, interpolate_missing, Normalizer
import warnings
import visibility_graph
import networkx as nx
from statsmodels.tsa.stattools import acf
from functools import partial
from utils.tools import worker_init_fn
# import random
# def worker_init_fn(worker_id,seed):
#     worker_seed = seed
#     random.seed(worker_seed)
#     np.random.seed(worker_seed)
#     torch.manual_seed(worker_seed)

class vgData_stock(Dataset):
    def __init__(self, flag="train",root_path="data/",data_path="600809.SH.csv",target="updown",scale=True,seq_len=20,pre_len=0,num_dims=6):
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.scale=scale
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.seq_len = seq_len
        self.set_type = type_map[flag]
        self.target=target
        self.num_dims=num_dims
        self.adj_cache=[]
        self.last100close=[]
        self.last100date=[]
        self.__read_data__()
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols= list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test 
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        featurers_data=df_raw[cols]
        if self.scale:
            train_data = featurers_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(featurers_data.values)
        else:
            data = featurers_data.values
        self.data_x = data[border1:border2]
        data_y=df_raw[[self.target]]
        self.data_y = data_y[border1:border2]
        if self.flag == 'test':
            # 提取测试集范围内的原始date和close数据
            test_range_df = df_raw.iloc[border1:border2]
            self.last100date = test_range_df['date'].tolist()[19:]       # 测试集日期列表
            self.last100close = test_range_df['close'].tolist()[19:]   # 测试集收盘价列表
            # print(f"测试集数据提取完成：日期{len(self.last100date)}条，收盘价{len(self.last100close)}条")
        #添加
        total_samples = len(self.data_x) - self.seq_len + 1  # 总样本数
        for i in range(total_samples):
            # 取第i个样本的序列（长度为seq_len）
            s_begin = i
            s_end = s_begin + self.seq_len
            seq_x = self.data_x[s_begin:s_end]  # 形状：(seq_len, num_dims)
            
            # 计算该样本每个维度的邻接矩阵
            sample_adj = []
            for dim in range(self.num_dims):
                dim_seq = seq_x[:, dim]  # 取第dim维的序列（长度为seq_len）
                adj = self.make_vg(dim_seq)  # 生成邻接矩阵
                # adj = self.make_acf(dim_seq) #生成acf邻接矩阵
                # adj = self.make_easy_adj(dim_seq) #生成单位矩阵
                sample_adj.append(adj)  # 每个维度的邻接矩阵：(seq_len, seq_len)
            
            # 将当前样本的所有维度邻接矩阵存入缓存
            # 形状：(num_dims, seq_len, seq_len)
            self.adj_cache.append(torch.stack(sample_adj))
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin+self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        label = self.data_y.iloc[s_end-1]
        adj_list=self.adj_cache[index]
        # adj_list=[]
        # for dim in range(self.num_dims):
        #     dim_seq = seq_x[:,dim]
        #     adj=self.make_vg(dim_seq)
        #     adj_list.append(adj)
        seq_x = torch.from_numpy(seq_x).float()  # 转为 float 张量（模型通常需要 float32）
        label = torch.tensor(label, dtype=torch.float32)  # 标签转为张量
        #adj_list=torch.stack(adj_list)

        return seq_x, adj_list,label  
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1     
    def make_vg(self,seq_x):
        vg_data=visibility_graph.visibility_graph(seq_x)
        adj_matrix=nx.to_numpy_array(vg_data, dtype=int)
        adj_matrix=adj_matrix+np.eye(20)
        adj=torch.from_numpy(adj_matrix).float()
        return adj
    def make_acf(self,seq_x):
        seq_len=len(seq_x)
        acf_values=acf(seq_x,nlags=seq_len-1,fft=False)
        adj_matrix=np.zeros((seq_len,seq_len),dtype=np.float32)
        for i in range(seq_len):
            for j in range(seq_len):
                k=abs(i-j)
                adj_matrix[i][j]=acf_values[k]
        adj_matrix=(np.abs(adj_matrix)>0.2).astype(np.float32)
        return torch.from_numpy(adj_matrix)
    """
    基于自相关函数(ACF)构建邻接矩阵
    seq_x: 单个维度的时间序列，形状为(seq_len,)
    返回: 形状为(seq_len, seq_len)的邻接矩阵（torch.Tensor）
    """

    def make_easy_adj(self,seq_x):
        seq_len=len(seq_x)
        adj_matrix=np.eye(seq_len)
        return torch.from_numpy(adj_matrix).to(torch.float32)


def data_provider(batch_size,flag,data_path):
    #shuffle_flag = False if (flag == 'test' or flag == 'val') else True
    Data = vgData_stock
    data_set = Data(flag=flag,data_path=data_path)
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False,
                             worker_init_fn=partial(worker_init_fn,seed=2025))
    return data_set, data_loader

if __name__ == '__main__':
    # vg,b=data_provider(batch_size=1,flag='val',data_path='000001.SZ.csv')
    # print(b.__len__())
    lxj=vgData_stock(flag="test")
    print(lxj.last100date)




    


    
    

