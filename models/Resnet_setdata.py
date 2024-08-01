

import os

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SetData():
    def __init__(self):
        self.data = None
        self.target = None
        self.data_path = None
        self.train_ds = None
        self.val_ds = None
        self.train_meanRGB = None
        self.train_stdRGB = None
        self.train_meanR = None
        self.train_meanG = None
        self.train_meanB = None
        self.train_stdR = None
        self.train_stdG = None
        self.train_stdB = None
        self.val_meanRGB = None
        self.val_stdRGB = None
        self.val_meanR = None
        self.val_meanG = None
        self.val_meanB = None
        self.val_stdR = None
        self.val_stdG = None
        self.val_stdB = None
        self.train_transformation = None
        self.val_transformation = None
        self.train_ds = None
        self.val_ds = None
        self.train_dl = None
        self.val_dl = None
    
    def download_data(self):
        self.data_path = './data'
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        # load dataset
        # self.train_ds = datasets.STL10(self.data_path, split='train', download=True, transform=transforms.ToTensor())
        # self.val_ds = datasets.STL10(self.data_path, split='test', download=True, transform=transforms.ToTensor())

        # self.train_ds = datasets.CIFAR10("./", train=True, transform=transforms.ToTensor(), download=True)
        # self.val_ds = datasets.CIFAR10("./", train=False, transform=transforms.ToTensor(), download=True)

        train_subset = datasets.CIFAR10("./", train=True, transform=transforms.ToTensor(), download=True)
        test_subset = datasets.CIFAR10("./", train=False, transform=transforms.ToTensor(), download=True)

        self.train_ds = torch.utils.data.Subset(train_subset, indices=range(int(len(train_subset) * 0.5)))
        self.val_ds = torch.utils.data.Subset(test_subset, indices=range(int(len(test_subset) * 0.5)))


        print(len(self.train_ds))
        print(len(self.val_ds))

    def nomalize_data(self):
        # To normalize the dataset, calculate the mean and std
        self.train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in self.train_ds]
        self.train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in self.train_ds]

        self.train_meanR = np.mean([m[0] for m in self.train_meanRGB])
        self.train_meanG = np.mean([m[1] for m in self.train_meanRGB])
        self.train_meanB = np.mean([m[2] for m in self.train_meanRGB])
        self.train_stdR = np.mean([s[0] for s in self.train_stdRGB])
        self.train_stdG = np.mean([s[1] for s in self.train_stdRGB])
        self.train_stdB = np.mean([s[2] for s in self.train_stdRGB])

        self.val_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in self.val_ds]
        self.val_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in self.val_ds]
        
        self.val_meanR = np.mean([m[0] for m in self.val_meanRGB])
        self.val_meanG = np.mean([m[1] for m in self.val_meanRGB])
        self.val_meanB = np.mean([m[2] for m in self.val_meanRGB])

        self.val_stdR = np.mean([s[0] for s in self.val_stdRGB])
        self.val_stdG = np.mean([s[1] for s in self.val_stdRGB])
        self.val_stdB = np.mean([s[2] for s in self.val_stdRGB])

        print(self.train_meanR, self.train_meanG, self.train_meanB)
        print(self.val_meanR, self.val_meanG, self.val_meanB)
    
    def set_transformation(self):
        # define the image transformation
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
        self.train_ds.transform = self.train_transformation
        self.val_ds.transform = self.val_transformation

        # create DataLoader
        self.train_dl = DataLoader(self.train_ds, batch_size=32, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=32, shuffle=True)
    
    def run(self):
        self.download_data()
        self.nomalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl
