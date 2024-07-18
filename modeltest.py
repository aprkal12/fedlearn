import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import CNN

class base_model:
    def __init__(self):
        self.model = None
        self.loss_func = None
        self.optimizer = None
        self.device = None
        self.train_loader = None
        self.test_loader = None

    def set_variable(self):
        cifar_train = dset.CIFAR10("./", train=True, transform=transforms.ToTensor(), download=True)
        cifar_test = dset.CIFAR10("./", train=False, transform=transforms.ToTensor(), download=True)

        # 학습 데이터의 10%만 사용
        # train_subset = torch.utils.data.Subset(cifar_train, indices=range(int(len(cifar_train) * 0.7)))
        # test_subset = torch.utils.data.Subset(cifar_test, indices=range(int(len(cifar_test) * 0.7)))

        # DataLoader 설정
        batch_size = 128
        self.train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)
        train_size = len(cifar_train)
        test_size = len(cifar_test)
        print("학습데이터 사이즈 :",train_size)
        print("테스트 데이터 사이즈 :",test_size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = CNN().to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    # 학습
    def train(self, device, model, train_loader, optimizer, loss_func):
        for epoch in range(5):
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
            accuracy = self.eval_accuracy(model, self.train_loader, self.device)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch+1, 5, i+1, len(train_loader), loss.item(), accuracy))

            # # 정확도가 90% 이상이면 학습 중지
            # if accuracy >= 90.0:
            #     print('Reached 90% accuracy. Stopping training.')
            #     break        

    def eval(self, device, model, test_loader):
        # 테스트
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('모델 테스트 결과 : {} %\n'.format(100 * correct / total))
    
    def eval_accuracy(self, model, data_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def parameter_extract(self, model):
        weights = model.state_dict()
        return weights

    def tensor_to_numpy(self, tensor):
        return tensor.cpu().detach().numpy().tolist()
    
    def get_base_model(self):
        return self.model

    def load_parameter(self, parameter):
        self.model.load_state_dict(parameter)

    def run(self):
        self.train(self.device, self.model, self.train_loader, self.optimizer, self.loss_func)
        self.eval(self.device, self.model, self.test_loader)
        # self.parameter_extract(self.model)


if __name__ == '__main__':
    model = base_model()
    model.run()