import os
from utils.tools import write_result , seed_everything ,save_epochacc
from argparse import Namespace
import torch
from train import vggcn_train
import random
import numpy as np
def onetrain(save_path:str,data_path:str):
    args= Namespace(
        use_gpu=1,
        gpu_type='cuda',
        gpu=0,
        device="",
        use_multi_gpu=False,
        train_epochs=2,
        checkpoints="train/",
        patience=10,
        learning_rate=0.001,
        batch_size=16,
        data_path=data_path,
        result_file_path=save_path,

    )
    seed=2025
    seed_everything(seed)
    if torch.cuda.is_available() and args.use_gpu:
        print(torch.cuda.get_device_name(0))
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print(args.device)
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')
    setting="first_train"
    exp=vggcn_train(args)
    setting="first_train"
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache() 

if __name__ == "__main__":
    save_path="pre_result.csv"
    write_result(save_path,flag=1)
    save_epochacc(file_name="VAL_pre_epoch.csv", flag=1)
    save_epochacc(file_name="test_pre_epoch.csv",flag=1)
    folder="data/"
    for csv_file in os.listdir(folder):
        onetrain(save_path,csv_file) 
        '''
        如何进行消融实验：
        1，对于vg我们使用acf自相关系数来代替visibility_graph生成邻接矩阵
            在data_loader.py中68，69行选择使用vg or 使用acf
        2，对于gcn模型，我们使用gatmodel来代替gcn
            在train中22行__buildmodel函数中选择使用gcn or gat
        '''