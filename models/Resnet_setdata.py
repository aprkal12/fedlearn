import os
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class SetData():
    def __init__(self):
        self.data_path = './data'
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.train_meanRGB = None
        self.train_stdRGB = None
        self.val_meanRGB = None
        self.val_stdRGB = None
        self.test_meanRGB = None
        self.test_stdRGB = None
        self.train_transformation = None
        self.val_transformation = None
        self.test_transformation = None
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
    
    def download_data(self, data_size):
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        train_dataset = datasets.CIFAR10(self.data_path, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR10(self.data_path, train=False, transform=transforms.ToTensor(), download=True)
        
        total_train_size = int(len(train_dataset) * data_size)
        train_indices = list(range(total_train_size))
        np.random.shuffle(train_indices)

        val_test_data_size = 10000 if total_train_size * 0.2 > 10000 else total_train_size * 0.2

        val_test_indices = list(range(int(val_test_data_size)))
        np.random.shuffle(val_test_indices)
        val_size = len(val_test_indices) // 2
        test_size = len(val_test_indices) - val_size

        val_indices = val_test_indices[:val_size]
        test_indices = val_test_indices[val_size:]

        self.train_ds = Subset(train_dataset, train_indices)
        self.val_ds = Subset(test_dataset, val_indices)
        self.test_ds = Subset(test_dataset, test_indices)

        print("Train set size:", len(self.train_ds))
        print("Validation set size:", len(self.val_ds))
        print("Test set size:", len(self.test_ds))
        
    def normalize_data(self):
        # Calculate mean and std for train dataset
        self.train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in self.train_ds]
        self.train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in self.train_ds]
        self.train_meanR = np.mean([m[0] for m in self.train_meanRGB])
        self.train_meanG = np.mean([m[1] for m in self.train_meanRGB])
        self.train_meanB = np.mean([m[2] for m in self.train_meanRGB])
        self.train_stdR = np.mean([s[0] for s in self.train_stdRGB])
        self.train_stdG = np.mean([s[1] for s in self.train_stdRGB])
        self.train_stdB = np.mean([s[2] for s in self.train_stdRGB])

        # Calculate mean and std for validation dataset
        self.val_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in self.val_ds]
        self.val_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in self.val_ds]
        self.val_meanR = np.mean([m[0] for m in self.val_meanRGB])
        self.val_meanG = np.mean([m[1] for m in self.val_meanRGB])
        self.val_meanB = np.mean([m[2] for m in self.val_meanRGB])
        self.val_stdR = np.mean([s[0] for s in self.val_stdRGB])
        self.val_stdG = np.mean([s[1] for s in self.val_stdRGB])
        self.val_stdB = np.mean([s[2] for s in self.val_stdRGB])

        # Calculate mean and std for test dataset
        self.test_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in self.test_ds]
        self.test_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in self.test_ds]
        self.test_meanR = np.mean([m[0] for m in self.test_meanRGB])
        self.test_meanG = np.mean([m[1] for m in self.test_meanRGB])
        self.test_meanB = np.mean([m[2] for m in self.test_meanRGB])
        self.test_stdR = np.mean([s[0] for s in self.test_stdRGB])
        self.test_stdG = np.mean([s[1] for s in self.test_stdRGB])
        self.test_stdB = np.mean([s[2] for s in self.test_stdRGB])

        print('Train mean:', self.train_meanR, self.train_meanG, self.train_meanB)
        print('Validation mean:', self.val_meanR, self.val_meanG, self.val_meanB)
        print('Test mean:', self.test_meanR, self.test_meanG, self.test_meanB)
    
    def set_transformation(self):
        self.train_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.Normalize([self.train_meanR, self.train_meanG, self.train_meanB],[self.train_stdR, self.train_stdG, self.train_stdB]),
            transforms.RandomHorizontalFlip(),
        ])

        self.val_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.Normalize([self.val_meanR, self.val_meanG, self.val_meanB],[self.val_stdR, self.val_stdG, self.val_stdB]),
        ])
        
        self.test_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.Normalize([self.test_meanR, self.test_meanG, self.test_meanB],[self.test_stdR, self.test_stdG, self.test_stdB]),
        ])

        self.train_ds.dataset.transform = self.train_transformation
        self.val_ds.dataset.transform = self.val_transformation
        self.test_ds.dataset.transform = self.test_transformation

        self.train_dl = DataLoader(self.train_ds, batch_size=32, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=32, shuffle=False)
        self.test_dl = DataLoader(self.test_ds, batch_size=32, shuffle=False)
    
    def run(self, data_size):
        self.download_data(data_size)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl

# # 실행 예시
# set_data = SetData()
# train_dl, val_dl, test_dl = set_data.run()
