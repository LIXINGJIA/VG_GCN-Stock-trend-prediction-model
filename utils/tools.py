import numpy as np
import torch
import math
import csv
import random
import os
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience # how many times will you tolerate for loss not being on decrease
        self.verbose = verbose  # whether to print tip info
        self.counter = 0 # now how many times loss not on decrease
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        # meaning: current score is not 'delta' better than best_score, representing that 
        # further training may not bring remarkable improvement in loss. 
        elif score < self.best_score + self.delta:  
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 'No Improvement' times become higher than patience --> Stop Further Training
            if self.counter >= self.patience:
                self.early_stop = True

        else: #model's loss is still on decrease, save the now best model and go on training
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
    ### used for saving the current best model
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def cal_precision(y_pred, y_true):
    tp = np.sum((y_pred == 1)& ((y_true == 1)))
    predicted_positive = np.sum(y_pred == 1)
    if predicted_positive == 0 :
        return 0.0
    return tp / predicted_positive

def cal_recall(y_pred, y_true):
    tp = np.sum((y_pred == 1)& ((y_true == 1)))
    actual_positive = np.sum(y_true == 1)
    if actual_positive == 0 :
        return 0.0
    return tp / actual_positive

def cal_f1(y_pred, y_true):
    precision = cal_precision(y_pred, y_true)
    recall = cal_recall(y_pred, y_true)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def write_result(file_name:str,stock=0,acc=0,pre=0,recall=0,f1=0,time=0,flag=1):
    if flag==1:
        with open(file_name,"a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["stocck", "acc", "precision", "recall", "f1","time"])
            print("write csv header")
    else:
        with open(file_name,"a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow([stock,acc,pre,recall,f1,time])
            print("write result to {}".format(file_name))


def save_epochacc(file_name: str, stock: str = None, epoch_acc_list: list = None, flag: int = 1):
    """
    保存股票在所有迭代轮次的acc值到CSV文件（epoch_acc_list为每个epoch的acc列表）
    
    参数:
        file_name: CSV文件路径
        stock: 股票名称（列名，flag=0时必需）
        epoch_acc_list: 包含每个epoch对应acc值的列表（flag=0时必需）
        flag: 1-初始化创建新文件；0-写入数据
    """
    if flag == 1:
        # 初始化模式：创建新文件并写入表头（仅包含epoch列）
        with open(file_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'])  # 第一列固定为迭代轮次
    else:
        # 写入数据模式：检查必要参数
        if stock is None or epoch_acc_list is None:
            raise ValueError("当flag=0时，stock和epoch_acc_list为必需参数")
        
        headers = []
        rows = []
        
        # 读取已有数据（如果文件存在）
        if os.path.exists(file_name):
            with open(file_name, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)  # 获取现有表头
                rows = [list(row) for row in reader]  # 读取所有行数据
        
        # 如果文件不存在或表头为空，初始化表头
        if not headers:
            headers = ['epoch']
        
        # 若当前股票不在表头中，添加新列
        if stock not in headers:
            headers.append(stock)
            # 为已有行补充空值以匹配新列数
            for row in rows:
                row.append('')
        
        # 总epoch数为列表长度（假设列表索引0对应epoch 1）
        total_epochs = len(epoch_acc_list)
        
        # 确保行数足够（若现有行数 < 总epoch数，补充空行）
        while len(rows) < total_epochs:
            new_row = [''] * len(headers)
            # 填充当前行的epoch值（第一列，从1开始）
            new_row[0] = str(len(rows) + 1)
            rows.append(new_row)
        
        # 填充当前股票的每个epoch acc值
        col_idx = headers.index(stock)
        for epoch_idx in range(total_epochs):
            row_idx = epoch_idx  # 列表索引0对应行索引0（epoch 1）
            # 确保该行的epoch值正确（防止之前的空行epoch未填充）
            rows[row_idx][0] = str(epoch_idx + 1)
            # 写入acc值（保留6位小数）
            rows[row_idx][col_idx] = f"{epoch_acc_list[epoch_idx]:.6f}"
        
        # 将更新后的数据写回文件
        with open(file_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # 写入表头
            writer.writerows(rows)    # 写入所有行数据

def seed_everything(seed=42):
    """
    Seed everything to make the experiment reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id,seed):
    worker_seed = seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def save_huice_results_to_csv(trues, predictions, dates, close_prices, save_path):
    """
    保存测试结果到CSV文件
    :param trues: 真实值列表
    :param predictions: 预测值列表
    :param dates: 日期列表
    :param close_prices: 收盘价列表
    :param save_path: 保存路径
    """
    # 确保所有列表长度一致
    min_length = min(len(trues), len(predictions), len(dates), len(close_prices))
    trues = trues[:min_length]
    predictions = predictions[:min_length]
    dates = dates[:min_length]
    close_prices = close_prices[:min_length]
        
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    # 写入CSV文件
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['日期', '收盘价', '真实值', '预测值'])
        # 写入数据行
        for date, close, true_val, pred_val in zip(dates, close_prices, trues, predictions):
            writer.writerow([date, close, true_val, pred_val])
        
    print(f"测试结果已保存到: {save_path}")

