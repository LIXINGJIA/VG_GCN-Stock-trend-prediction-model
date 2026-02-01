import torch
import torch.nn as nn
from torch_geometric.nn import GATConv  # GAT层
from torch_geometric.utils import dense_to_sparse
class GATModel(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32, num_heads=4, seq_len=20):
        super(GATModel, self).__init__()
        # 第一层GAT：多注意力头（num_heads），输出拼接
        self.seq_len=seq_len
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,  # 注意力头数（关键参数）
            concat=True  # 多头输出是否拼接（第一层通常拼接）
        )
        # 第二层GAT：单头输出（与预测维度匹配）
        self.gat2 = GATConv(
            in_channels=hidden_channels * num_heads,  # 输入维度=第一层输出拼接后的维度
            out_channels=hidden_channels,
            heads=1,  # 单头
            concat=False  # 不拼接（直接输出）
        )
        self.relu=nn.ReLU()
        self.fc=nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x,adj):
        adj=adj[:,1,:,:].squeeze(1)
        batch_size=x.shape[0]
        edge_index_list = []
        for b in range(batch_size):
            ei, _ = dense_to_sparse(adj[b])
            edge_index_list.append(ei + b * self.seq_len)  # 偏移节点索引以区分不同batch
        
        edge_index = torch.cat(edge_index_list, dim=1)
        x=x.view(-1,x.shape[-1])
        # x: (num_nodes, in_channels)，节点特征
        # edge_index: (2, num_edges)，图的边索引（替代邻接矩阵）
        x = self.gat1(x, edge_index)  # 输出: (num_nodes, hidden_channels * num_heads)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.gat2(x, edge_index)  
        x=self.relu(x)
        x=x.view(batch_size,self.seq_len,-1)
        x=x[:,-1,:]
        x=self.fc(x)
        x=torch.sigmoid(x)
        return x
    


    
    
if __name__ == '__main__':
    adj=torch.randn(3,6,20,20)
    print(adj.shape)
    adj=adj[:,1,:,:].squeeze(1)
    print(adj.shape)
    edge_index_list = []
    edge_weight_list = []
    for b in range(3):
        ei, ew = dense_to_sparse(adj[b])
        edge_index_list.append(ei + b * 20)  # 偏移节点索引以区分不同batch
        edge_weight_list.append(ew)
    print(edge_index_list)