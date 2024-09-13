import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import copy

from models.Resnet_blocks import BasicBlock
from models.Resnet_mainblock import ResNet, resnet18, resnet50
from models.Resnet_setdata import SetData

# from Resnet_blocks import BasicBlock
# from Resnet_mainblock import ResNet, resnet18, resnet50
# from Resnet_setdata import SetData

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Inference():
    def __init__(self):
        self.model = None
        self.loss_func = None
        self.optimizer = None
        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.lr_scheduler = None
        self.params_train = None
        self.loss_hist = None
        self.metric_hist = None
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.best_model_wts = None
        self.epoch = None
        self.model_name = None
        self.loss_hist = None
        self.metric_hist = None

    # def split_non_iid_data(self, num_clients, data_size):
    #     SetData().split_non_iid_imbalanced_data(num_clients, data_size)
    
    # def split_non_iid_data100(self, num_clients, data_size):
    #     SetData().split_non_iid_100(num_clients, data_size)

    # def split_client_data(self, num_clients, data_size):
    #     SetData().split_client_data(num_clients, data_size)
    
    # def check_data(self, num_clients):
    #     SetData().check_imbalanced_data_distribution(num_clients)
    
    # def check_cifar100_data(self, num_clients):
    #     SetData().check_imbalanced_data_100(num_clients)

    def set_variable(self, data_size=None, client_id=None, non_iid_set=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = resnet18().to(self.device)
        self.model_name = 'resnet18'
        # self.model = resnet50().to(self.device)

        if client_id is not None and non_iid_set is None:
            self.train_dl, self.val_dl, self.test_dl = SetData().client_run(client_id)
        elif client_id is not None and non_iid_set is not None:
            self.train_dl, self.val_dl, self.test_dl = SetData().non_iid_client_run(client_id)
        elif non_iid_set is not None:
            self.train_dl, self.val_dl, self.test_dl = SetData().run100(data_size)
        else:
            self.train_dl, self.val_dl, self.test_dl = SetData().run(data_size)
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        # self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        # 스케쥴러 -> 10 에폭동안 val_loss가 줄어들지 않으면 lr을 0.1배로 줄인다.

    # lr 계산
    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']
    
    # 배치당 loss와 metric 계산
    def metric_batch(self, output, target):
        pred = output.argmax(1, keepdim=True)
        corrects = pred.eq(target.view_as(pred)).sum().item()
        return corrects
    
    def loss_batch(self, loss_func, output, target, opt=None):
        loss = loss_func(output, target)
        metric_b = self.metric_batch(output, target)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        return loss.item(), metric_b

    # 에폭당 loss
    def loss_epoch(self, model, loss_func, dataset_dl, sanity_check=False, opt=None):
        running_loss = 0.0
        running_metric = 0.0
        len_data = len(dataset_dl.dataset)

        for xb, yb in dataset_dl:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            output = model(xb)

            loss_b, metric_b = self.loss_batch(loss_func, output, yb, opt)

            running_loss += loss_b
            
            if metric_b is not None:
                running_metric += metric_b
            
            if sanity_check is True:
                break

        loss = running_loss / len_data
        metric = running_metric / len_data

        return loss, metric
    
    def train_val(self, model, params):
        num_epochs=params['num_epochs']
        loss_func=params["loss_func"]
        opt=params["optimizer"]
        train_dl=params["train_dl"]
        val_dl=params["val_dl"]
        test_dl=params["test_dl"]
        sanity_check=params["sanity_check"]
        lr_scheduler=params["lr_scheduler"]
        path2weights=params["path2weights"]

        self.loss_hist = {'train': [], 'val': [], 'test': []}
        self.metric_hist = {'train': [], 'val': [], 'test': []}

        best_loss = float('inf')

        start_time = time.time()
        best_epoch = 0

        early_stop = 0

        for epoch in range(num_epochs):
            early_stop += 1
            current_lr = self.get_lr(opt)
            print('Epoch {}/{}, current lr={}'.format(epoch+1, num_epochs, current_lr))

            model.train()
            train_loss, train_metric = self.loss_epoch(model, loss_func, train_dl, sanity_check, opt)
            self.loss_hist['train'].append(train_loss)
            self.metric_hist['train'].append(train_metric)

            model.eval()
            with torch.no_grad():
                val_loss, val_metric = self.loss_epoch(model, loss_func, val_dl, sanity_check)
            self.loss_hist['val'].append(val_loss)
            self.metric_hist['val'].append(val_metric)

            model.eval()
            with torch.no_grad():
                test_loss, test_metric = self.loss_epoch(model, loss_func, test_dl, sanity_check)
            self.loss_hist['test'].append(test_loss)
            self.metric_hist['test'].append(test_metric)

            if val_loss < best_loss:
                early_stop = 0
                best_loss = val_loss
                self.best_model_wts = model.state_dict()
                best_epoch = epoch+1
                print('Get best val_loss')
            
            # lr_scheduler.step(val_loss)

            print("train loss: %.6f, val loss: %.6f, accuracy: %.2f %%, time: %.4f min" %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
            print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {100*test_metric:.2f}%")

            if early_stop > 20: # 얼리스탑 커맨드
                print("early stop!!!")
                print('-'*10)
                break

        print("best epoch : ", best_epoch)
        self.early_stop_epoch = epoch + 1  # Update early stop epoch

        return model, self.loss_hist, self.metric_hist

    
    def get_params_file(self):
        torch.save(self.parameter_extract(), 'models\\weights.pt')
        return 'models\\weights.pt'
    
    def get_accuracy(self, model, mode = 'None'):
        if mode == 'train':
            return self.loss_hist['train'][-1], self.metric_hist['train'][-1]
        elif mode == 'val':
            model.eval()
            with torch.no_grad():
                val_loss, val_metric = self.loss_epoch(model, self.loss_func, self.val_dl, False)
            return val_loss, val_metric
        elif mode == 'test':
            model.eval()
            with torch.no_grad():
                test_loss, test_metric = self.loss_epoch(model, self.loss_func, self.test_dl, False)
            return test_loss, test_metric
        else:
            return None
    
    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error')
    
    def fill_graph(self):
        # Train-Validation Progress
        num_epochs = self.early_stop_epoch if self.early_stop_epoch is not None else self.params_train["num_epochs"]

        # plot loss progress
        plt.title("Train-Val Loss")
        plt.plot(range(1, num_epochs + 1), self.loss_hist["train"], label="train")
        plt.plot(range(1, num_epochs + 1), self.loss_hist["val"], label="val")
        plt.ylabel("Loss")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        # plot accuracy progress
        plt.title("Train-Val Accuracy")
        plt.plot(range(1, num_epochs + 1), self.metric_hist["train"], label="train")
        plt.plot(range(1, num_epochs + 1), self.metric_hist["val"], label="val")
        plt.ylabel("Accuracy")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()
    
    def parameter_extract(self):
        return self.model.state_dict()
    
    def tensor_to_numpy(self, tensor):
        return tensor.cpu().detach().numpy().tolist()
    
    def load_parameter(self, parameter):
        self.model.load_state_dict(parameter)
    
    def set_epoch(self, n):
        self.epoch = n
    
    def get_data_size(self):
        data = {
            'train': len(self.train_dl.dataset),
            'val': len(self.val_dl.dataset),
            'test': len(self.test_dl.dataset)
        }
        return data
    
    def run(self):
        self.params_train = {
            'num_epochs': self.epoch,
            'optimizer': self.optimizer,
            'loss_func': self.loss_func,
            'train_dl': self.train_dl,
            'val_dl': self.val_dl,
            'test_dl': self.test_dl,
            'sanity_check': False,
            'lr_scheduler': self.lr_scheduler,
            'path2weights': './models/weights.pt',
        }
        self.createFolder('./models')

        self.model, self.loss_hist, self.metric_hist = self.train_val(self.model, self.params_train)

        
        # self.fill_graph()  # 학습 및 검증 결과 그래프 출력

if __name__ == '__main__':
    infer = Inference()
    infer.set_variable(1) # 사용할 데이터 사이즈 (0 ~ 1) 비율로 설정
    infer.set_epoch(50)
    # summary(infer.model, (3, 32, 32))
    # print(infer.parameter_extract())
    infer.run()
