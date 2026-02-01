import torch 
import torch.nn as nn
import torch.nn.functional as F


# 移除原有GCN类，无需图卷积操作

class vg_fcn_model(nn.Module):
    def __init__(self, num_f=6, seq_len=20):
        super().__init__()
        self.seq_len = seq_len    # 时间窗口长度（20）
        self.num_f = num_f        # 特征数量（6）
        
        # 计算扁平化后的总输入维度：
        # 每个特征：节点特征(seq_len*1) + 邻接矩阵(seq_len*seq_len) = 20 + 400 = 420
        # 6个特征总维度：420 * 6 = 2520
        self.flatten_dim = num_f * (seq_len + seq_len * seq_len)
        
        # 全连接网络（适配高维扁平化特征，保留dropout防止过拟合）
        self.fn1 = nn.Linear(self.flatten_dim, 128)  # 第一层降维
        self.fn2 = nn.Linear(128, 64)                # 第二层降维
        self.fn3 = nn.Linear(64, 1)                  # 输出层（涨跌预测）
        self.dropout = nn.Dropout(0.2)               # 与原模型一致的dropout率

    def forward(self, data, f_list):
        # data: (batch_size, seq_len, num_f) - 节点特征（原始时序数据）
        # f_list: (batch_size, num_f, seq_len, seq_len) - VG生成的邻接矩阵
        
        batch_size = data.shape[0]
        flatten_features = []  # 存储每个特征的扁平化结果
        
        # 遍历每个特征，处理VG生成的图结构（节点特征+邻接矩阵扁平化）
        for i in range(self.num_f):
            # 1. 提取第i个特征的节点特征：(batch_size, seq_len)
            node_feat = data[:, :, i]  # (B, 20)
            # 2. 提取第i个特征的VG邻接矩阵：(batch_size, seq_len, seq_len)
            adj = f_list[:, i, :, :]   # (B, 20, 20)
            
            # 3. 扁平化：节点特征(20) + 邻接矩阵(400) → 420维/特征
            adj_flat = adj.view(batch_size, -1)  # 邻接矩阵扁平化：(B, 400)
            feat_flat = torch.cat([node_feat, adj_flat], dim=1)  # (B, 420)
            
            flatten_features.append(feat_flat)
        
        # 4. 拼接所有特征的扁平化结果：(batch_size, 2520)
        cat_out = torch.cat(flatten_features, dim=1)
        
        # 5. 全连接网络前向传播
        x = self.fn1(cat_out)
        x = F.relu(x)          # ReLU激活
        x = self.dropout(x)    # dropout防止过拟合
        
        x = self.fn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fn3(x)
        x = torch.sigmoid(x)   # 输出0-1概率（涨跌）
        
        return x

if __name__ == '__main__':
    # 测试参数与原代码一致，确保兼容性
    batch_size = 3
    seq_len = 20
    num_f = 6    
    
    # 初始化仅VG+FCN的模型（无GCN）
    model = vg_fcn_model(
        num_f=num_f,
        seq_len=seq_len
    )
    
    # 构造测试输入（与原代码格式完全一致）
    data = torch.randn(batch_size, seq_len, num_f)    # (3, 20, 6) 节点特征
    f_list = torch.randn(batch_size, num_f, seq_len, seq_len)  # (3,6,20,20) VG邻接矩阵
    
    # 前向传播测试
    output = model(data, f_list)
    print("输出形状（应等于(batch_size, 1)）：", output.shape)  # 预期: (3, 1)