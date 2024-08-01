
# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt
from torchsummary import summary

# utils
import numpy as np
import time
import copy

from torch.optim.lr_scheduler import ReduceLROnPlateau

from Resnet_blocks import BasicBlock
from Resnet_mainblock import ResNet, resnet18
from Resnet_mainblock import resnet50
from Resnet_setdata import SetData

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# wandb.init(
#     project="Federated Learning",
#     entity="aprkal12",
#     config={
#         "learning_rate": 0.001,
#         "architecture": "Resnet18",
#         "dataset": "CIFAR-10",
#     }
# )
# wandb.run.name = "Resnet18_CIFAR-10_B=100%"

class Inference():
    def __init__(self):
        self.model = None
        self.loss_func = None
        self.optimizer = None
        self.device = None
        self.train_loader = None
        self.test_loader = None
        self.lr_scheduler = None
        self.params_train = None
        self.loss_hist = None
        self.metric_hist = None
        self.train_dl = None
        self.val_dl = None
        self.best_model_wts = None
        self.epoch = None

    
    def set_variable(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = resnet18().to(self.device)
        # self.model = resnet50().to(self.device)

        self.train_dl, self.val_dl = SetData().run()
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)

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
        sanity_check=params["sanity_check"]
        lr_scheduler=params["lr_scheduler"]
        path2weights=params["path2weights"]

        loss_history = {'train': [], 'val': []}
        metric_history = {'train': [], 'val': []}

        # # GPU out of memoty error
        # best_model_wts = copy.deepcopy(model.state_dict())

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
            loss_history['train'].append(train_loss)
            metric_history['train'].append(train_metric)

            model.eval()
            with torch.no_grad():
                val_loss, val_metric = self.loss_epoch(model, loss_func, val_dl, sanity_check)
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

            if val_loss < best_loss:
                early_stop = 0
                best_loss = val_loss
                self.best_model_wts = model.state_dict()

                # torch.save(model.state_dict(), path2weights)
                # print('Copied best model weights!')
                best_epoch = epoch+1
                print('Get best val_loss')
            

            lr_scheduler.step(val_loss)

            print("train loss: %.6f, val loss: %.6f, accuracy: %.2f %%, time: %.4f min" %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
            print('-'*10)
            # wandb.log({"train loss":train_loss, "val loss":val_loss, "accuracy": val_metric})
            if early_stop > 20:
                print("early stop!!!")
                print('-'*10)
                break

        print("best epoch : ", best_epoch)
        # model.load_state_dict(best_model_wts)

        return model, loss_history, metric_history
    
    def get_params_file(self):
        torch.save(self.parameter_extract(), 'models\\weights.pt')
        return 'models\\weights.pt'
    
    def get_accuracy(self, model):
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = self.loss_epoch(model, self.params_train["loss_func"], self.params_train["val_dl"], self.params_train["sanity_check"])
        return val_loss, val_metric
    
    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error')
    
    def fill_graph(self):
        # Train-Validation Progress
        num_epochs=self.params_train["num_epochs"]

        # plot loss progress
        plt.title("Train-Val Loss")
        plt.plot(range(1,num_epochs+1),self.loss_hist["train"],label="train")
        plt.plot(range(1,num_epochs+1),self.loss_hist["val"],label="val")
        plt.ylabel("Loss")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        # plot accuracy progress
        plt.title("Train-Val Accuracy")
        plt.plot(range(1,num_epochs+1),self.metric_hist["train"],label="train")
        plt.plot(range(1,num_epochs+1),self.metric_hist["val"],label="val")
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
    
    def run(self):
        self.params_train = {
            'num_epochs':self.epoch,
            'optimizer':self.optimizer,
            'loss_func':self.loss_func,
            'train_dl':self.train_dl,
            'val_dl':self.val_dl,
            'sanity_check':False,
            'lr_scheduler':self.lr_scheduler,
            'path2weights':'./models/weights.pt',
        }
        self.createFolder('./models')

        self.model, self.loss_hist, self.metric_hist = self.train_val(self.model, self.params_train)

if __name__ == '__main__':
    infer = Inference()
    infer.set_variable()
    infer.set_epoch(1)
    # summary(infer.model, (3, 32, 32))
    # print(infer.parameter_extract())
    infer.run()
    