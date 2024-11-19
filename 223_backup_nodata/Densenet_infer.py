import json
import zstd
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Resnet_setdata import DataManager
from Densenet import DenseNet121, DenseNet264
import utils  # 제공된 DenseNet 모델을 사용

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class InferenceDenseNet:
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
        self.best_model_wts = None
        self.epoch = None

    def set_variable(self, data_size=None, client_id=None, non_iid_set=None, num_clients=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = DenseNet264().to(self.device)

        data_manager = DataManager()

        if client_id is not None and non_iid_set is False:
            self.train_dl, self.val_dl, self.test_dl = data_manager.client_run(client_id, num_clients)
        elif client_id is not None and non_iid_set is not False:
            self.train_dl, self.val_dl, self.test_dl = data_manager.non_iid_client_run(client_id, num_clients)
        elif non_iid_set is not None:
            data_manager.dataset_name = 'CIFAR100'
            self.train_dl, self.val_dl, self.test_dl = data_manager.run(data_size)
        else:
            self.train_dl, self.val_dl, self.test_dl = data_manager.run(data_size)

        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']

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
        num_epochs = params['num_epochs']
        loss_func = params["loss_func"]
        opt = params["optimizer"]
        train_dl = params["train_dl"]
        val_dl = params["val_dl"]
        sanity_check = params["sanity_check"]
        path2weights = params["path2weights"]

        self.loss_hist = {'train': [], 'val': []}
        self.metric_hist = {'train': [], 'val': []}

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

            if val_loss < best_loss:
                early_stop = 0
                best_loss = val_loss
                self.best_model_wts = model.state_dict()
                best_epoch = epoch + 1
                print('Get best val_loss')

            print("train loss: {:.6f}, val loss: {:.6f}, accuracy: {:.2f}%, time: {:.4f} min".format(
                train_loss, val_loss, 100 * val_metric, (time.time() - start_time) / 60))

            if early_stop > 20:  # Early stopping condition
                print("early stop!!!")
                break

        print("best epoch : ", best_epoch)
        return model, self.loss_hist, self.metric_hist

    def run(self):
        self.params_train = {
            'num_epochs': self.epoch,
            'optimizer': self.optimizer,
            'loss_func': self.loss_func,
            'train_dl': self.train_dl,  # self.train_loader -> self.train_dl
            'val_dl': self.val_dl,      # self.val_loader -> self.val_dl
            'sanity_check': False,
            'path2weights': './models/weights.pt',
        }
        self.model, self.loss_hist, self.metric_hist = self.train_val(self.model, self.params_train)

if __name__ == '__main__':
    infer = InferenceDenseNet()
    infer.set_variable(1)
    infer.epoch = 1

    # torchsummary.summary(infer.model, (3, 32, 32))
    # 1. 텐서를 넘파이 배열로 변환 (먼저 CUDA 텐서를 CPU로 복사)
    parameters = infer.model.state_dict()  # 예시로 파라미터 추출
    parameters_as_numpy = {k: v.cpu().numpy() for k, v in parameters.items()}  # .cpu() 추가

    torch.save(parameters, 'densenet121')
    serial = pickle.dumps(parameters)
    print("serialized = ", utils.bytes_to_mb(len(serial)))
    # 2. 넘파이 배열을 JSON으로 직렬화하려면 넘파이 배열을 리스트로 변환해야 함
    parameters_as_list = {k: v.tolist() for k, v in parameters_as_numpy.items()}

    # 3. JSON 형식으로 직렬화
    json_data = json.dumps(parameters_as_list)

    print("JSON = ", utils.bytes_to_mb(len(json_data)))

    ############# 파라미터 데이터 타입을 가볍게 해서 전송용 파라미터 사이즈 감소
    ############# bfloat16을 사용하면 float32랑 지수부는 같고 가수부에서 사이즈감소
    ############# float16 대신 bfloat16 쓰면 오버플로우 방지된다 함
    infer.run()

    data = infer.model.to(torch.bfloat16)
    halfdata = infer.model.state_dict()
    # halfdata.to(torch.bfloat16)
    bytedata = pickle.dumps(halfdata)
    zstddata = zstd.compress(bytedata)
    print("제안 방식 파라미터 사이즈 : %.2f MB" % utils.bytes_to_mb(len(zstddata)))
    
    # dedata = zstd.decompress(zstddata)
    # debyte = pickle.loads(dedata)
    # debyte.float()
    # infer.model.load_state_dict(debyte)
    # # infer.model.float()

    # infer.run()
    # #############

    # infer.set_variable(1)  # 데이터셋 설정
    # infer.epoch = 10  # 학습 에폭 설정
    # infer.run()
