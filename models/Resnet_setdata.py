import os
import pickle
import random
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class SetData():
    def __init__(self):
        self.data_path = './data'
        self.client_data_dir = './client_data'
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
    
    def split_client_data(self, num_clients, data_size):
        self.client_data_dir = './client_data'
        if not os.path.exists(self.client_data_dir):
            os.makedirs(self.client_data_dir)

        # 데이터 다운로드 및 준비
        train_dataset = datasets.CIFAR10(self.data_path, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR10(self.data_path, train=False, transform=transforms.ToTensor(), download=True)
        
        total_train_size = int(len(train_dataset) * data_size)
        indices = list(range(total_train_size))
        np.random.shuffle(indices)

        # Validation/Test 데이터셋을 위한 인덱스 설정
        val_test_data_size = min(10000, int(total_train_size * 0.2))
        val_test_indices = list(range(val_test_data_size))
        np.random.shuffle(val_test_indices)
        val_size = len(val_test_indices) // 2
        test_size = len(val_test_indices) - val_size
        val_indices = val_test_indices[:val_size]
        test_indices = val_test_indices[val_size:]

        # 클라이언트별 데이터셋 나누기
        client_train_data_size = total_train_size // num_clients
        client_val_size = val_size // num_clients
        client_test_size = test_size // num_clients

        for i in range(num_clients):
            # Train 데이터 나누기
            start_idx = i * client_train_data_size
            end_idx = (i + 1) * client_train_data_size if i != num_clients - 1 else total_train_size
            client_train_indices = indices[start_idx:end_idx]
            client_train_dataset = Subset(train_dataset, client_train_indices)
            
            # Validation/Test 데이터 나누기
            start_val_idx = i * client_val_size
            end_val_idx = (i + 1) * client_val_size if i != num_clients - 1 else val_size
            client_val_indices = val_indices[start_val_idx:end_val_idx]
            client_val_dataset = Subset(test_dataset, client_val_indices)

            start_test_idx = i * client_test_size
            end_test_idx = (i + 1) * client_test_size if i != num_clients - 1 else test_size
            client_test_indices = test_indices[start_test_idx:end_test_idx]
            client_test_dataset = Subset(test_dataset, client_test_indices)

            # 데이터 파일로 저장
            with open(os.path.join(self.client_data_dir, f'0.5_client_{i}_train_data.pkl'), 'wb') as f:
                pickle.dump(client_train_dataset, f)
            with open(os.path.join(self.client_data_dir, f'0.5_client_{i}_val_data.pkl'), 'wb') as f:
                pickle.dump(client_val_dataset, f)
            with open(os.path.join(self.client_data_dir, f'0.5_client_{i}_test_data.pkl'), 'wb') as f:
                pickle.dump(client_test_dataset, f)

        print(f"Client data saved to {self.client_data_dir}")
    
    def split_non_iid_imbalanced_data(self, num_clients, data_size):
        self.client_data_dir = './client_data_non_iid_imbalanced'
        if not os.path.exists(self.client_data_dir):
            os.makedirs(self.client_data_dir)

        # 데이터 다운로드 및 준비
        train_dataset = datasets.CIFAR10(self.data_path, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR10(self.data_path, train=False, transform=transforms.ToTensor(), download=True)

        total_train_size = int(len(train_dataset) * data_size)
        indices = list(range(total_train_size))
        np.random.shuffle(indices)

        # 각 클래스별 데이터셋 분리
        class_data = {i: [] for i in range(10)}
        for idx in indices:
            _, label = train_dataset[idx]
            class_data[label].append(idx)

        # 클라이언트별 데이터셋 초기화
        client_train_data = {i: [] for i in range(num_clients)}
        client_val_data = {i: [] for i in range(num_clients)}
        client_test_data = {i: [] for i in range(num_clients)}

        # 각 클래스별로 데이터 분배
        for class_id, class_list in class_data.items():
            num_samples = len(class_list)
            random.shuffle(class_list)

            # 데이터 나누기
            for i in range(num_clients):
                num_client_samples = num_samples // num_clients
                start_idx = i * num_client_samples
                end_idx = (i + 1) * num_client_samples if i != num_clients - 1 else num_samples
                client_data = class_list[start_idx:end_idx]

                # 데이터 추가
                client_train_data[i].extend(client_data[:int(len(client_data) * 0.8)])
                client_val_data[i].extend(client_data[int(len(client_data) * 0.8):int(len(client_data) * 0.9)])
                client_test_data[i].extend(client_data[int(len(client_data) * 0.9):])

        # 데이터 파일로 저장
        for i in range(num_clients):
            with open(os.path.join(self.client_data_dir, f'client_{i}_train_data_non_iid_imbalanced.pkl'), 'wb') as f:
                pickle.dump(Subset(train_dataset, client_train_data[i]), f)
            with open(os.path.join(self.client_data_dir, f'client_{i}_val_data_non_iid_imbalanced.pkl'), 'wb') as f:
                pickle.dump(Subset(train_dataset, client_val_data[i]), f)
            with open(os.path.join(self.client_data_dir, f'client_{i}_test_data_non_iid_imbalanced.pkl'), 'wb') as f:
                pickle.dump(Subset(test_dataset, client_test_data[i]), f)

        print(f"Client data saved to {self.client_data_dir}")

    def check_imbalanced_data_distribution(self, num_clients):
        self.client_data_dir = './client_data_non_iid_imbalanced'
        for client_id in range(num_clients):
            print(f"\nClient {client_id} data distribution:")

            # Train 데이터 분포 확인
            with open(os.path.join(self.client_data_dir, f'client_{client_id}_train_data_non_iid_imbalanced.pkl'), 'rb') as f:
                client_train_data = pickle.load(f)
            train_class_counts = [0] * 10
            for _, label in client_train_data:
                train_class_counts[label] += 1
            print(f"  Train class distribution: {train_class_counts}")

            # Validation 데이터 분포 확인
            with open(os.path.join(self.client_data_dir, f'client_{client_id}_val_data_non_iid_imbalanced.pkl'), 'rb') as f:
                client_val_data = pickle.load(f)
            val_class_counts = [0] * 10
            for _, label in client_val_data:
                val_class_counts[label] += 1
            print(f"  Validation class distribution: {val_class_counts}")

            # Test 데이터 분포 확인
            with open(os.path.join(self.client_data_dir, f'client_{client_id}_test_data_non_iid_imbalanced.pkl'), 'rb') as f:
                client_test_data = pickle.load(f)
            test_class_counts = [0] * 10
            for _, label in client_test_data:
                test_class_counts[label] += 1
            print(f"  Test class distribution: {test_class_counts}")


    def load_non_iid_client_data(client_id):
        data_dir = './client_data_non_iid_imbalanced'
        
        with open(os.path.join(data_dir, f'client_{client_id}_train_data.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        
        with open(os.path.join(data_dir, f'client_{client_id}_val_data.pkl'), 'rb') as f:
            val_data = pickle.load(f)
        
        with open(os.path.join(data_dir, f'client_{client_id}_test_data.pkl'), 'rb') as f:
            test_data = pickle.load(f)

        return train_data, val_data, test_data


    def load_client_data(self, client_id):
        train_file = os.path.join(self.client_data_dir, f'0.5_client_{client_id}_train_data.pkl')
        val_file = os.path.join(self.client_data_dir, f'0.5_client_{client_id}_val_data.pkl')
        test_file = os.path.join(self.client_data_dir, f'0.5_client_{client_id}_test_data.pkl')

        with open(train_file, 'rb') as f:
            self.train_ds = pickle.load(f)
        with open(val_file, 'rb') as f:
            self.val_ds = pickle.load(f)
        with open(test_file, 'rb') as f:
            self.test_ds = pickle.load(f)

        print("client_id:", client_id)
        print("Train set size:", len(self.train_ds))
        print("Validation set size:", len(self.val_ds))
        print("Test set size:", len(self.test_ds))

        print("Client data loaded")
    # # 예제 사용법
    # save_client_data('data', num_clients=5, data_size=1.0, self.client_data_dir='client_data')

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
    
    def client_run(self, client_id):
        self.load_client_data(client_id)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl
    
    def run(self, data_size):
        self.download_data(data_size)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl

# # 실행 예시
# set_data = SetData()
# train_dl, val_dl, test_dl = set_data.run()
