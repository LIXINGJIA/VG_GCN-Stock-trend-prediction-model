import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import time
from gcn_model import vg_gcn_model #use GCN_model
from gat_model import GATModel #use GAT_model
from vg_fcn_model import vg_fcn_model
from data_provider.data_loader import data_provider
from utils.tools import EarlyStopping,cal_accuracy,cal_f1,cal_precision,cal_recall,write_result,save_epochacc,save_huice_results_to_csv


class vggcn_train():
    total_time =0
    def __init__(self,args):
        self.args=args
        self.device=self._acquire_device()
        self.model=self._build_model().to(self.device)
    def _build_model(self):
        train_data,train_loader = self._get_data(flag="train")
        test_data, test_loader = self._get_data(flag='test')
        model=vg_gcn_model(num_f=6,seq_len=20,hidden_dim1=1,hidden_dim2=1,hidden_dim3=1)
        # model=GATModel()#选择model 
        # model = vg_fcn_model(num_f=6, seq_len=20)
        return model


    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    def _get_data(self,flag):
        data_set, data_loader = data_provider(batch_size=self.args.batch_size,flag=flag,data_path=self.args.data_path)
        return data_set,data_loader
    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.BCELoss()
        return criterion    
    def vali(self,vali_data,vali_loader,criterion):
        total_loss=[]
        preds=[]
        trues=[]
        self.model.eval()
        with torch.no_grad():
            for i , (batch_x,adj_list, label) in enumerate(vali_loader):
                batch_x=batch_x.to(self.device)
                adj_list=adj_list.float().to(self.device)
                label=label.to(self.device)
                outputs= self.model(batch_x,adj_list)
                pred = outputs.detach()
                loss=criterion(pred, label.float())
                total_loss.append(loss.item())
                preds.append(outputs.detach())
                trues.append(label)
        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        predictions=(preds>0.5).float().cpu().numpy().flatten()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss ,accuracy



    def train(self,setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag='val')#改为小写，val之前为TEST
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        epochs=[]
        epochs_2=[]
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        begin_time = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            
            epoch_time = time.time()

            for i, (batch_x, adj_list,label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                adj_list=adj_list.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x,adj_list)
                loss = criterion(outputs, label.float())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            epochs.append(val_accuracy)
            epochs_2.append(test_accuracy)
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        self.total_time = time.time() - begin_time
        print("prediction_total_time: {}".format(self.total_time))
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        save_epochacc(file_name="VAL_pre_epoch.csv",stock=self.args.data_path[:-4],epoch_acc_list=epochs,flag=0)
        save_epochacc(file_name="test_pre_epoch.csv",stock=self.args.data_path[:-4],epoch_acc_list=epochs_2,flag=0)
        return self.model
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')#改为小写
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, adj_list,label) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                adj_list=adj_list.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, adj_list)

                preds.append(outputs.detach())
                trues.append(label)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)
        predictions=(preds>0.5).float().cpu().numpy().flatten()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        precision = cal_precision(predictions, trues)
        recall = cal_recall(predictions, trues)
        f1 = cal_f1(predictions, trues)

        print("真实值：",trues,"预测值",predictions)
        # print("日期",test_data.last100date)
        # print("close price",test_data.last100close)
        # result save
        save_huice_results_to_csv(trues=trues,predictions=predictions,dates=test_data.last100date,close_prices=test_data.last100close,save_path="huicedata/"+self.args.data_path)


        # test_path = folder_path + 'test_results.csv'    
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        print('precision:{}'.format(precision))
        print('recall:{}'.format(recall))
        print('f1:{}'.format(f1))
        write_result(file_name=self.args.result_file_path,stock=self.args.data_path[:-4],acc=accuracy,pre=precision,recall=recall,f1=f1,time=self.total_time,flag=0)