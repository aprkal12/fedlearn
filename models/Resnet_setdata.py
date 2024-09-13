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

    # def split_non_iid_100(self, num_clients, data_size):
    #     """
    #     각 클래스의 분포는 임의로 다르게 설정하되, 전체 데이터셋의 8:1:1 비율을 유지하며 클라이언트에 데이터셋을 분배합니다.
    #     CIFAR-100에 맞춘 코드입니다.
        
    #     :param num_clients: 클라이언트 수
    #     :param data_size: 전체 데이터 크기 비율 (0~1 사이의 값)
    #     """
    #     self.client_data_dir = './client_data_custom_distribution'
    #     if not os.path.exists(self.client_data_dir):
    #         os.makedirs(self.client_data_dir)

    #     # CIFAR-100 데이터 다운로드 및 준비
    #     train_dataset = datasets.CIFAR100(self.data_path, train=True, transform=transforms.ToTensor(), download=True)
    #     test_dataset = datasets.CIFAR100(self.data_path, train=False, transform=transforms.ToTensor(), download=True)

    #     total_train_size = int(len(train_dataset) * data_size)
    #     indices = list(range(total_train_size))
    #     np.random.shuffle(indices)

    #     # 각 클래스별로 데이터셋 분리 (100개 클래스)
    #     class_data = {i: [] for i in range(100)}  # CIFAR-100은 100개의 클래스
    #     for idx in indices:
    #         _, label = train_dataset[idx]
    #         class_data[label].append(idx)

    #     # 클라이언트별 데이터셋 초기화
    #     client_train_data = {i: [] for i in range(num_clients)}
    #     client_val_data = {i: [] for i in range(num_clients)}
    #     client_test_data = {i: [] for i in range(num_clients)}

    #     # 각 클래스별로 데이터를 8:1:1 비율로 정확히 분배
    #     for class_id, class_list in class_data.items():
    #         random.shuffle(class_list)
    #         class_size = len(class_list)

    #         # 전체 데이터를 먼저 8:1:1 비율로 나눕니다
    #         num_train = int(0.8 * class_size)
    #         num_val = int(0.1 * class_size)
    #         num_test = class_size - num_train - num_val

    #         train_data = class_list[:num_train]
    #         val_data = class_list[num_train:num_train + num_val]
    #         test_data = class_list[num_train + num_val:]

    #         # 각 클라이언트에 임의의 비율로 데이터 분배
    #         train_proportions = np.random.dirichlet(np.ones(num_clients))
    #         val_proportions = np.random.dirichlet(np.ones(num_clients))
    #         test_proportions = np.random.dirichlet(np.ones(num_clients))

    #         train_start_idx = 0
    #         val_start_idx = 0
    #         test_start_idx = 0
            
    #         for i in range(num_clients):
    #             # 각 클라이언트별로 할당된 비율에 맞춰 데이터를 분배
    #             train_samples = int(train_proportions[i] * num_train)
    #             val_samples = int(val_proportions[i] * num_val)
    #             test_samples = int(test_proportions[i] * num_test)

    #             # 데이터 할당
    #             client_train_data[i].extend(train_data[train_start_idx:train_start_idx + train_samples])
    #             client_val_data[i].extend(val_data[val_start_idx:val_start_idx + val_samples])
    #             client_test_data[i].extend(test_data[test_start_idx:test_start_idx + test_samples])

    #             # 인덱스 갱신
    #             train_start_idx += train_samples
    #             val_start_idx += val_samples
    #             test_start_idx += test_samples

    #     # 데이터를 파일로 저장
    #     for i in range(num_clients):
    #         with open(os.path.join(self.client_data_dir, f'client_{i}_train_data_custom_distribution.pkl'), 'wb') as f:
    #             pickle.dump(Subset(train_dataset, client_train_data[i]), f)
    #         with open(os.path.join(self.client_data_dir, f'client_{i}_val_data_custom_distribution.pkl'), 'wb') as f:
    #             pickle.dump(Subset(train_dataset, client_val_data[i]), f)
    #         with open(os.path.join(self.client_data_dir, f'client_{i}_test_data_custom_distribution.pkl'), 'wb') as f:
    #             pickle.dump(Subset(test_dataset, client_test_data[i]), f)

    #     print(f"Client data saved to {self.client_data_dir}")
        
    def split_non_iid_imbalanced_data(self, num_clients, data_size, test_ratio=0.1, val_ratio=0.1):
        self.client_data_dir = './client_data_non_iid_imbalanced'
        if not os.path.exists(self.client_data_dir):
            os.makedirs(self.client_data_dir)

        # 데이터 다운로드 및 준비
        train_dataset = datasets.CIFAR10(self.data_path, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR10(self.data_path, train=False, transform=transforms.ToTensor(), download=True)

        total_train_size = int(len(train_dataset) * data_size)
        indices = list(range(total_train_size))
        np.random.shuffle(indices)

        # 각 클래스별로 데이터셋 분리
        class_data = {i: [] for i in range(10)}
        for idx in indices:
            _, label = train_dataset[idx]
            class_data[label].append(idx)

        # 테스트 데이터 분배
        test_class_data = {i: [] for i in range(10)}
        for idx in range(len(test_dataset)):
            _, label = test_dataset[idx]
            test_class_data[label].append(idx)

        client_train_data = {i: [] for i in range(num_clients)}
        client_val_data = {i: [] for i in range(num_clients)}
        client_test_data = {i: [] for i in range(num_clients)}

        # 각 클래스별로 임의의 비율로 데이터를 클라이언트에 분배
        for class_id, class_list in class_data.items():
            random.shuffle(class_list)
            class_size = len(class_list)

            # 각 클라이언트에 무작위 비율로 데이터 분배 (0~1 사이 비율)
            proportions = np.random.dirichlet(np.ones(num_clients))
            start_idx = 0
            for i in range(num_clients):
                num_client_samples = int(proportions[i] * class_size)
                client_data = class_list[start_idx:start_idx + num_client_samples]
                start_idx += num_client_samples

                # 각 클라이언트 내에서 8:1:1 비율로 train, validation, test 나누기
                num_val = int(val_ratio * len(client_data))
                num_train = len(client_data) - num_val

                client_train_data[i].extend(client_data[:num_train])
                client_val_data[i].extend(client_data[num_train:])

        # 각 클라이언트에 테스트 데이터 분배
        for class_id, class_list in test_class_data.items():
            random.shuffle(class_list)
            proportions = np.random.dirichlet(np.ones(num_clients))  # 무작위 비율로 테스트 데이터 분배
            start_idx = 0
            for i in range(num_clients):
                num_client_samples = int(proportions[i] * len(class_list))
                client_test_data[i].extend(class_list[start_idx:start_idx + num_client_samples])
                start_idx += num_client_samples

        # 데이터를 파일로 저장
        for i in range(num_clients):
            with open(os.path.join(self.client_data_dir, f'client_{i}_train_data_non_iid_imbalanced.pkl'), 'wb') as f:
                pickle.dump(Subset(train_dataset, client_train_data[i]), f)
            with open(os.path.join(self.client_data_dir, f'client_{i}_val_data_non_iid_imbalanced.pkl'), 'wb') as f:
                pickle.dump(Subset(train_dataset, client_val_data[i]), f)
            with open(os.path.join(self.client_data_dir, f'client_{i}_test_data_non_iid_imbalanced.pkl'), 'wb') as f:
                pickle.dump(Subset(test_dataset, client_test_data[i]), f)

        print(f"Client data saved to {self.client_data_dir}")

    # def check_imbalanced_data_100(self, num_clients):
    #     self.client_data_dir = './client_data_custom_distribution'
        
    #     for client_id in range(num_clients):
    #         print(f"\nClient {client_id} data distribution:")

    #         # Train 데이터 분포 및 데이터 개수 확인
    #         with open(os.path.join(self.client_data_dir, f'client_{client_id}_train_data_custom_distribution.pkl'), 'rb') as f:
    #             client_train_data = pickle.load(f)
    #         train_class_counts = [0] * 100  # CIFAR-100은 100개의 클래스
    #         for _, label in client_train_data:
    #             train_class_counts[label] += 1
    #         print(f"  Train class distribution: {train_class_counts}")
    #         print(f"  Total train samples: {sum(train_class_counts)}")

    #         # Validation 데이터 분포 및 데이터 개수 확인
    #         with open(os.path.join(self.client_data_dir, f'client_{client_id}_val_data_custom_distribution.pkl'), 'rb') as f:
    #             client_val_data = pickle.load(f)
    #         val_class_counts = [0] * 100  # CIFAR-100은 100개의 클래스
    #         for _, label in client_val_data:
    #             val_class_counts[label] += 1
    #         print(f"  Validation class distribution: {val_class_counts}")
    #         print(f"  Total validation samples: {sum(val_class_counts)}")

    #         # Test 데이터 분포 및 데이터 개수 확인
    #         with open(os.path.join(self.client_data_dir, f'client_{client_id}_test_data_custom_distribution.pkl'), 'rb') as f:
    #             client_test_data = pickle.load(f)
    #         test_class_counts = [0] * 100  # CIFAR-100은 100개의 클래스
    #         for _, label in client_test_data:
    #             test_class_counts[label] += 1
    #         print(f"  Test class distribution: {test_class_counts}")
    #         print(f"  Total test samples: {sum(test_class_counts)}")

    def check_imbalanced_data_distribution(self, num_clients):
        self.client_data_dir = './client_data_non_iid_imbalanced'
        for client_id in range(num_clients):
            print(f"\nClient {client_id} data distribution:")

            # Train 데이터 분포 및 데이터 개수 확인
            with open(os.path.join(self.client_data_dir, f'client_{client_id}_train_data_non_iid_imbalanced.pkl'), 'rb') as f:
                client_train_data = pickle.load(f)
            train_class_counts = [0] * 10
            for _, label in client_train_data:
                train_class_counts[label] += 1
            print(f"  Train class distribution: {train_class_counts}")
            print(f"  Total train samples: {sum(train_class_counts)}")

            # Validation 데이터 분포 및 데이터 개수 확인
            with open(os.path.join(self.client_data_dir, f'client_{client_id}_val_data_non_iid_imbalanced.pkl'), 'rb') as f:
                client_val_data = pickle.load(f)
            val_class_counts = [0] * 10
            for _, label in client_val_data:
                val_class_counts[label] += 1
            print(f"  Validation class distribution: {val_class_counts}")
            print(f"  Total validation samples: {sum(val_class_counts)}")

            # Test 데이터 분포 및 데이터 개수 확인
            with open(os.path.join(self.client_data_dir, f'client_{client_id}_test_data_non_iid_imbalanced.pkl'), 'rb') as f:
                client_test_data = pickle.load(f)
            test_class_counts = [0] * 10
            for _, label in client_test_data:
                test_class_counts[label] += 1
            print(f"  Test class distribution: {test_class_counts}")
            print(f"  Total test samples: {sum(test_class_counts)}")


    def load_non_iid_client_data(self, client_id):
        print(f"cifar-100 client data {client_id}")
        data_dir = './cifar100_non_iid'
        
        with open(os.path.join(data_dir, f'client_{client_id}_train_data.pkl'), 'rb') as f:
            self.train_ds = pickle.load(f)
        
        with open(os.path.join(data_dir, f'client_{client_id}_val_data.pkl'), 'rb') as f:
            self.val_ds = pickle.load(f)
        
        with open(os.path.join(data_dir, f'client_{client_id}_test_data.pkl'), 'rb') as f:
            self.test_ds = pickle.load(f)

        print("client_id:", client_id)
        print("Train set size:", len(self.train_ds))
        print("Validation set size:", len(self.val_ds))
        print("Test set size:", len(self.test_ds))

        print("Client data loaded")


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

    def download_100_data(self, data_size):
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        # CIFAR-100 데이터셋 로드
        train_dataset = datasets.CIFAR100(self.data_path, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR100(self.data_path, train=False, transform=transforms.ToTensor(), download=True)
        
        total_train_size = int(len(train_dataset) * data_size)
        train_indices = list(range(total_train_size))
        np.random.shuffle(train_indices)

        # Validation과 Test 데이터를 나누는 부분 (데이터 크기를 10,000으로 제한)
        val_test_data_size = 10000 if total_train_size * 0.2 > 10000 else total_train_size * 0.2

        val_test_indices = list(range(int(val_test_data_size)))
        np.random.shuffle(val_test_indices)
        val_size = len(val_test_indices) // 2
        test_size = len(val_test_indices) - val_size

        val_indices = val_test_indices[:val_size]
        test_indices = val_test_indices[val_size:]

        # 데이터셋을 Subset으로 나누기
        self.train_ds = Subset(train_dataset, train_indices)
        self.val_ds = Subset(test_dataset, val_indices)
        self.test_ds = Subset(test_dataset, test_indices)

        print("Train set size:", len(self.train_ds))
        print("Validation set size:", len(self.val_ds))
        print("Test set size:", len(self.test_ds))

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
    
    def non_iid_client_run(self, client_id):
        self.load_non_iid_client_data(client_id)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl

    def client_run(self, client_id):
        self.load_client_data(client_id)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl
    
    def run100(self, data_size):
        self.download_100_data(data_size)
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
