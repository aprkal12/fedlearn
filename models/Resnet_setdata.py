import os
import pickle
import random
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class DataManager:
    def __init__(self, dataset_name='CIFAR10', data_path='./data', client_data_dir='./for_client_data'):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.client_data_dir = client_data_dir
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.train_mean = None
        self.train_std = None
        self.val_mean = None
        self.val_std = None
        self.test_mean = None
        self.test_std = None
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None

    def download_data(self, data_size):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        if self.dataset_name == 'CIFAR10':
            DatasetClass = datasets.CIFAR10
            num_classes = 10
        elif self.dataset_name == 'CIFAR100':
            DatasetClass = datasets.CIFAR100
            num_classes = 100
        else:
            raise ValueError("지원하지 않는 데이터셋입니다.")

        train_dataset = DatasetClass(self.data_path, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = DatasetClass(self.data_path, train=False, transform=transforms.ToTensor(), download=True)

        total_train_size = int(len(train_dataset) * data_size)
        train_indices = list(range(total_train_size))
        np.random.shuffle(train_indices)

        val_test_size = int(min(10000, total_train_size * 0.2))
        val_test_indices = list(range(val_test_size))
        np.random.shuffle(val_test_indices)
        val_size = val_test_size // 2
        test_size = val_test_size - val_size

        val_indices = val_test_indices[:val_size]
        test_indices = val_test_indices[val_size:]

        self.train_ds = Subset(train_dataset, train_indices)
        self.val_ds = Subset(test_dataset, val_indices)
        self.test_ds = Subset(test_dataset, test_indices)

        print("Train set size:", len(self.train_ds))
        print("Validation set size:", len(self.val_ds))
        print("Test set size:", len(self.test_ds))

    def compute_mean_std(self, dataset):
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
        mean = 0.0
        std = 0.0
        nb_samples = 0
        for data in loader:
            batch_samples = data[0].size(0)
            data = data[0].view(batch_samples, data[0].size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        return mean.numpy(), std.numpy()

    def normalize_data(self):
        self.train_mean, self.train_std = self.compute_mean_std(self.train_ds)
        self.val_mean, self.val_std = self.compute_mean_std(self.val_ds)
        self.test_mean, self.test_std = self.compute_mean_std(self.test_ds)

        print('Train mean:', self.train_mean)
        print('Train std:', self.train_std)
        print('Validation mean:', self.val_mean)
        print('Validation std:', self.val_std)
        print('Test mean:', self.test_mean)
        print('Test std:', self.test_std)

    def set_transformation(self):
        self.train_transforms = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(self.train_mean, self.train_std),
            transforms.RandomHorizontalFlip(),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(self.val_mean, self.val_std),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(self.test_mean, self.test_std),
        ])

        self.train_ds.dataset.transform = self.train_transforms
        self.val_ds.dataset.transform = self.val_transforms
        self.test_ds.dataset.transform = self.test_transforms

        self.train_dl = DataLoader(self.train_ds, batch_size=32, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=32, shuffle=False)
        self.test_dl = DataLoader(self.test_ds, batch_size=32, shuffle=False)

    def run(self, data_size):
        self.download_data(data_size)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl

    def run100(self, data_size):
        self.dataset_name = 'CIFAR100'
        return self.run(data_size)

    def split_data(self, num_clients, data_size, iid=True, test_ratio=0.1, val_ratio=0.1):
        if not os.path.exists(self.client_data_dir):
            os.makedirs(self.client_data_dir)

        self.download_data(data_size)
        if iid:
            # IID 데이터 분할
            total_train_size = len(self.train_ds)
            indices = list(range(total_train_size))
            np.random.shuffle(indices)
            client_data_size = total_train_size // num_clients

            for i in range(num_clients):
                start_idx = i * client_data_size
                end_idx = (i + 1) * client_data_size if i != num_clients - 1 else total_train_size
                client_indices = indices[start_idx:end_idx]
                client_train_dataset = Subset(self.train_ds.dataset, client_indices)

                # 클라이언트 데이터 저장
                with open(os.path.join(self.client_data_dir, f'client_{i}_train_data.pkl'), 'wb') as f:
                    pickle.dump(client_train_dataset, f)

                # 모든 클라이언트가 동일한 검증 및 테스트 데이터를 사용
                with open(os.path.join(self.client_data_dir, f'client_{i}_val_data.pkl'), 'wb') as f:
                    pickle.dump(self.val_ds, f)
                with open(os.path.join(self.client_data_dir, f'client_{i}_test_data.pkl'), 'wb') as f:
                    pickle.dump(self.test_ds, f)
        else:
            # Non-IID 데이터 분할
            num_classes = 10 if self.dataset_name == 'CIFAR10' else 100

            class_data = {i: [] for i in range(num_classes)}
            for idx in range(len(self.train_ds)):
                _, label = self.train_ds[idx]
                class_data[label].append(idx)

            client_train_data = {i: [] for i in range(num_clients)}
            client_val_data = {i: [] for i in range(num_clients)}
            client_test_data = {i: [] for i in range(num_clients)}

            for class_id, class_indices in class_data.items():
                random.shuffle(class_indices)
                class_size = len(class_indices)

                # Dirichlet 분포를 사용하여 클래스 데이터를 클라이언트에 분배
                proportions = np.random.dirichlet(np.ones(num_clients))
                start_idx = 0
                for i in range(num_clients):
                    num_samples = int(proportions[i] * class_size)
                    client_indices = class_indices[start_idx:start_idx + num_samples]
                    start_idx += num_samples

                    # 클라이언트 데이터 내에서 학습 및 검증 데이터로 분할
                    num_val = int(val_ratio * len(client_indices))
                    num_train = len(client_indices) - num_val

                    client_train_data[i].extend(client_indices[:num_train])
                    client_val_data[i].extend(client_indices[num_train:])

            # 테스트 데이터 분할
            test_class_data = {i: [] for i in range(num_classes)}
            for idx in range(len(self.test_ds)):
                _, label = self.test_ds[idx]
                test_class_data[label].append(idx)

            for class_id, class_indices in test_class_data.items():
                random.shuffle(class_indices)
                proportions = np.random.dirichlet(np.ones(num_clients))
                start_idx = 0
                for i in range(num_clients):
                    num_samples = int(proportions[i] * len(class_indices))
                    client_indices = class_indices[start_idx:start_idx + num_samples]
                    start_idx += num_samples

                    client_test_data[i].extend(client_indices)

            # 클라이언트 데이터 저장
            for i in range(num_clients):
                client_train_dataset = Subset(self.train_ds.dataset, client_train_data[i])
                client_val_dataset = Subset(self.train_ds.dataset, client_val_data[i])
                client_test_dataset = Subset(self.test_ds.dataset, client_test_data[i])

                with open(os.path.join(self.client_data_dir, f'client_{i}_train_data_non_iid.pkl'), 'wb') as f:
                    pickle.dump(client_train_dataset, f)
                with open(os.path.join(self.client_data_dir, f'client_{i}_val_data_non_iid.pkl'), 'wb') as f:
                    pickle.dump(client_val_dataset, f)
                with open(os.path.join(self.client_data_dir, f'client_{i}_test_data_non_iid.pkl'), 'wb') as f:
                    pickle.dump(client_test_dataset, f)

    def client_run(self, client_id):
        self.load_client_data(client_id, iid=True)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl

    def non_iid_client_run(self, client_id):
        self.load_client_data(client_id, iid=False)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl

    def load_client_data(self, client_id, iid=True):
        if iid:
            train_file = os.path.join(self.client_data_dir, f'client_{client_id}_train_data.pkl')
            val_file = os.path.join(self.client_data_dir, f'client_{client_id}_val_data.pkl')
            test_file = os.path.join(self.client_data_dir, f'client_{client_id}_test_data.pkl')
        else:
            train_file = os.path.join(self.client_data_dir, f'client_{client_id}_train_data_non_iid.pkl')
            val_file = os.path.join(self.client_data_dir, f'client_{client_id}_val_data_non_iid.pkl')
            test_file = os.path.join(self.client_data_dir, f'client_{client_id}_test_data_non_iid.pkl')

        with open(train_file, 'rb') as f:
            self.train_ds = pickle.load(f)
        with open(val_file, 'rb') as f:
            self.val_ds = pickle.load(f)
        with open(test_file, 'rb') as f:
            self.test_ds = pickle.load(f)

        print("Client ID:", client_id)
        print("Train set size:", len(self.train_ds))
        print("Validation set size:", len(self.val_ds))
        print("Test set size:", len(self.test_ds))
        print("클라이언트 데이터 로드 완료")

    def check_data_distribution(self, num_clients, iid=True):
        num_classes = 10 if self.dataset_name == 'CIFAR10' else 100
        for client_id in range(num_clients):
            print(f"\nClient {client_id} 데이터 분포:")

            self.load_client_data(client_id, iid=iid)
            # 학습 데이터 클래스 분포 확인
            class_counts = [0] * num_classes
            for _, label in self.train_ds:
                class_counts[label] += 1
            print(f"Train class distribution: {class_counts}")
            print(f"Total train samples: {sum(class_counts)}")

            # 검증 및 테스트 데이터도 동일하게 확인 가능
