from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
    # model=finalpredict_model(
    #             nfeat=1,
    #             nhid=1,
    #             nclass=1,
    #             dropout=0.2)
class GCN(nn.Module):
    def __init__(self,seq_len,hidden_dim1,hidden_dim2,hidden_dim3):
        super().__init__()
        self.seq_len=seq_len
        self.conv1=GCNConv(in_channels=1,out_channels=hidden_dim1)
        self.conv2=GCNConv(in_channels=hidden_dim1,
        out_channels=hidden_dim2)
        self.conv3=GCNConv(in_channels=hidden_dim2,
        out_channels=hidden_dim3)
        self.reshape_shape=seq_len*hidden_dim3
        self.dropout=nn.Dropout(0.2)
    def forward(self, x, adj,edge_weight=None):
 # x: (batch_size, seq_len, 1)
        # adj: (batch_size, seq_len, seq_len)
        batch_size = x.shape[0]
        
        # 批量转换稠密邻接矩阵为稀疏格式 (edge_index: [2, total_edges], edge_weight: [total_edges])
        edge_index_list = []
        edge_weight_list = []
        for b in range(batch_size):
            ei, ew = dense_to_sparse(adj[b])
            edge_index_list.append(ei + b * self.seq_len)  # 偏移节点索引以区分不同batch
            edge_weight_list.append(ew)
        
        # 合并所有batch的边信息
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_weight = torch.cat(edge_weight_list, dim=0)
        
        # 展平batch维度进行批量处理
        x = x.view(-1, 1)  # (batch_size * seq_len, 1)
        
        x = self.conv1(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        
        # 恢复batch维度
        x = x.view(batch_size, self.reshape_shape)
        return x




        # out=[]
        
        # for b in range(x.shape[0]):
        #     x_b=x[b]
        #     adj_b=adj[b]
        #     edge_index, edge_weight = dense_to_sparse(adj_b)
        #     x_b=self.conv1(x_b,edge_index,edge_weight)
        #     x_b=self.dropout(x_b)
        #     x_b=self.conv2(x_b,edge_index,edge_weight)
        #     x_b=F.relu(x_b)
        #     x_b=self.conv3(x_b,edge_index,edge_weight)
        #     x_b=x_b.reshape(self.reshape_shape)
        #     out.append(x_b)
        # return torch.stack(out,dim=0)
    

class vg_gcn_model(nn.Module):
    def __init__(self,num_f,seq_len,hidden_dim1,hidden_dim2,hidden_dim3):
        super().__init__()
        self.seq_len=seq_len
        self.num_f=num_f
        self.gcns=nn.ModuleList({GCN(seq_len=seq_len,hidden_dim1=hidden_dim1,hidden_dim2=hidden_dim2,hidden_dim3=hidden_dim3) for _ in range(num_f)})
        self.fn1=nn.Linear(120,60)
        self.fn2=nn.Linear(60,1)

    def forward(self,data,f_list):
         # data: (batch_size, seq_len, num_f)
         # f_list: (num_f, batch_size, seq_len, seq_len)
        # data=data.squeeze(dim=0)
        # f_list=f_list.squeeze(dim=0)
        dim_out=[]
        for i in range(self.num_f):
            f_data=data[:,:,i].unsqueeze(2)
            adj = f_list[:, i, :, :]
            out=self.gcns[i](f_data,adj)
            dim_out.append(out)
        cat_out=torch.cat(dim_out,dim=1)
        x=self.fn1(cat_out)
        x=self.fn2(x)
        x=torch.sigmoid(x)
        return x



if __name__=='__main__':
    batch_size = 3
    seq_len = 20
    num_f = 6    
    hidden_dim1 = 4
    hidden_dim2 = 2
    hidden_dim3 = 1
    
    # 初始化模型
    model = vg_gcn_model(
        num_f=num_f,
        seq_len=seq_len,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        hidden_dim3=hidden_dim3
    )
    data = torch.randn(batch_size, seq_len, num_f)  # (3, 20, 6)
    f_list = torch.randn(batch_size, num_f,seq_len, seq_len)  # (3,6,20,20)的邻接矩阵
    
    # 前向传播
    output = model(data, f_list)
    print("输出形状（应等于batch_size）：", output.shape)  # 预期: (3, 1)
