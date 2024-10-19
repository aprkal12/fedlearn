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
        if not os.path.exists(f'{self.client_data_dir}_{num_clients}'):
            os.makedirs(f'{self.client_data_dir}_{num_clients}')

        self.download_data(data_size)
        num_classes = 10 if self.dataset_name == 'CIFAR10' else 100
        distribution_type = "iid" if iid else "non_iid"

        # 전체 데이터를 섞어줍니다.
        train_indices = list(range(len(self.train_ds)))
        test_indices = list(range(len(self.test_ds)))
        val_indices = list(range(len(self.val_ds)))

        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        np.random.shuffle(val_indices)

        # # 섞인 인덱스에 해당하는 클래스 레이블을 가져옵니다.
        # train_labels = [self.train_ds.dataset.targets[idx] for idx in train_indices]
        # test_labels = [self.test_ds.dataset.targets[idx] for idx in test_indices]
        # val_labels = [self.val_ds.dataset.targets[idx] for idx in val_indices]


        # from collections import Counter 
        # # 클래스별 데이터 수를 카운트
        # train_class_count = Counter(train_labels)
        # test_class_count = Counter(test_labels)
        # val_class_count = Counter(val_labels)
        # # 클래스별 데이터 수 출력 및 총합 계산
        # print("Train dataset class counts:")
        # train_total_count = sum(train_class_count.values())
        # for class_label, count in train_class_count.items():
        #     print(f"Class {class_label}: {count} samples")
        # print(f"Total train samples: {train_total_count}")

        # print("\nTest dataset class counts:")
        # test_total_count = sum(test_class_count.values())
        # for class_label, count in test_class_count.items():
        #     print(f"Class {class_label}: {count} samples")
        # print(f"Total test samples: {test_total_count}")

        # print("\nValidation dataset class counts:")
        # val_total_count = sum(val_class_count.values())
        # for class_label, count in val_class_count.items():
        #     print(f"Class {class_label}: {count} samples")
        # print(f"Total validation samples: {val_total_count}")
        if iid:
            self._split_iid(num_clients, train_indices, test_indices, val_indices, distribution_type)
        else:
            self._split_non_iid(num_clients, train_indices, test_indices, val_indices, distribution_type)

        # self.print_dataset_statistics(num_clients, distribution_type)

    def _split_iid(self, num_clients, train_indices, test_indices, val_indices, distribution_type):
        # 1. train 데이터를 클라이언트 수대로 나눕니다.
        num_train_per_client = len(train_indices) // num_clients  # 각 클라이언트가 받을 train 데이터 크기
        clients_train_indices = [train_indices[i*num_train_per_client:(i+1)*num_train_per_client] for i in range(num_clients)]
         # 2. 각 클라이언트에 할당된 train 데이터의 0.1%를 validation과 test 데이터로 설정합니다.

        for i, client_train_indices in enumerate(clients_train_indices):
            client_train_size = len(client_train_indices)  # 각 클라이언트의 train 데이터 수
            val_size = int(client_train_size * 0.1)  # 0.1%의 데이터 크기
            test_size = int(client_train_size * 0.1)  # 0.1%의 데이터 크기

            # 3. test_indices에서 test 데이터 추출
            client_test_indices = test_indices[:test_size]
            test_indices = test_indices[test_size:]  # 사용된 test 인덱스는 제외

            # 4. val_indices에서 validation 데이터 추출
            client_val_indices = val_indices[:val_size]
            val_indices = val_indices[val_size:]  # 사용된 val 인덱스는 제외

            # 5. Subset을 만들어 각 클라이언트에 할당된 데이터를 저장합니다.
            client_train = Subset(self.train_ds.dataset, client_train_indices)
            client_val = Subset(self.train_ds.dataset, client_val_indices)
            client_test = Subset(self.train_ds.dataset, client_test_indices)

            # 6. 할당된 데이터를 저장 (필요한 방식에 맞춰 저장)
            self._save_client_data(i, client_train, client_val, client_test, distribution_type, num_clients)

            print(f"클라이언트 {i+1}: train = {len(client_train_indices)}, val = {len(client_val_indices)}, test = {len(client_test_indices)}")

    def _split_non_iid(self, num_clients, train_indices, test_indices, val_indices, distribution_type):
        total_train_samples = len(train_indices)

        # 각 클라이언트에 할당될 비율을 랜덤하게 생성 (합이 1이 되도록)
        proportions = np.random.dirichlet(np.ones(num_clients), size=1)[0]  # 디리클레 분포 사용 (합계가 1이 되도록 보장)

        # 1. train 데이터를 클라이언트 수대로 나누되, 각 클라이언트에 임의의 비율로 분배
        clients_train_indices = []
        start_idx = 0
        for i, proportion in enumerate(proportions):
            num_train_per_client = int(proportion * total_train_samples)  # 각 클라이언트에 할당할 데이터 수
            end_idx = start_idx + num_train_per_client
            clients_train_indices.append(train_indices[start_idx:end_idx])
            start_idx = end_idx

         # 2. 각 클라이언트에 할당된 train 데이터의 0.1%를 validation과 test 데이터로 설정합니다.

        for i, client_train_indices in enumerate(clients_train_indices):
            client_train_size = len(client_train_indices)  # 각 클라이언트의 train 데이터 수
            val_size = int(client_train_size * 0.1)  # 0.1%의 데이터 크기
            test_size = int(client_train_size * 0.1)  # 0.1%의 데이터 크기

            # 3. test_indices에서 test 데이터 추출
            client_test_indices = test_indices[:test_size]
            test_indices = test_indices[test_size:]  # 사용된 test 인덱스는 제외

            # 4. val_indices에서 validation 데이터 추출
            client_val_indices = val_indices[:val_size]
            val_indices = val_indices[val_size:]  # 사용된 val 인덱스는 제외

            # 5. Subset을 만들어 각 클라이언트에 할당된 데이터를 저장합니다.
            client_train = Subset(self.train_ds.dataset, client_train_indices)
            client_val = Subset(self.train_ds.dataset, client_val_indices)
            client_test = Subset(self.train_ds.dataset, client_test_indices)

            # 6. 할당된 데이터를 저장 (필요한 방식에 맞춰 저장)
            self._save_client_data(i, client_train, client_val, client_test, distribution_type, num_clients)

            print(f"클라이언트 {i+1}: train = {len(client_train_indices)}, val = {len(client_val_indices)}, test = {len(client_test_indices)}")

    def _save_client_data(self, client_id, train_data, val_data, test_data, distribution_type, num_clients=3):
        with open(os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{client_id}_train_data_{distribution_type}.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{client_id}_val_data_{distribution_type}.pkl'), 'wb') as f:
            pickle.dump(val_data, f)
        with open(os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{client_id}_test_data_{distribution_type}.pkl'), 'wb') as f:
            pickle.dump(test_data, f)

    def print_dataset_statistics(self, num_clients, distribution_type):
        total_train, total_val, total_test = 0, 0, 0
        for i in range(num_clients):
            print(f"\nClient {i} 데이터셋 통계:")
            for dataset_type in ['train', 'val', 'test']:
                with open(os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{i}_{dataset_type}_data_{distribution_type}.pkl'), 'rb') as f:
                    dataset = pickle.load(f)
                
                class_distribution = {}
                for idx in dataset.indices:
                    _, label = dataset.dataset[idx]
                    if label not in class_distribution:
                        class_distribution[label] = 0
                    class_distribution[label] += 1
                
                print(f"  {dataset_type.capitalize()} 셋:")
                print(f"    총 샘플 수: {len(dataset)}")
                print(f"    클래스별 분포:")
                for label, count in sorted(class_distribution.items()):
                    print(f"      클래스 {label}: {count}")

                if dataset_type == 'train':
                    total_train += len(dataset)
                elif dataset_type == 'val':
                    total_val += len(dataset)
                else:
                    total_test += len(dataset)

        print("\n전체 데이터셋 통계:")
        print(f"  Train 셋 총 샘플 수: {total_train}")
        print(f"  Val 셋 총 샘플 수: {total_val}")
        print(f"  Test 셋 총 샘플 수: {total_test}")

    def client_run(self, client_id, num_clients, distribution_type='iid'):
        self.load_client_data(client_id, num_clients, distribution_type, iid=True)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl

    def non_iid_client_run(self, client_id, num_clients, distribution_type='non_iid'):
        self.load_client_data(client_id, num_clients, distribution_type, iid=False)
        self.normalize_data()
        self.set_transformation()
        return self.train_dl, self.val_dl, self.test_dl

    def load_client_data(self, client_id, num_clients, distribution_type, iid=True):
        if iid:
            train_file = os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{client_id}_train_data_{distribution_type}.pkl')
            val_file = os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{client_id}_val_data_{distribution_type}.pkl')
            test_file = os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{client_id}_test_data_{distribution_type}.pkl')
        else:
            train_file = os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{client_id}_train_data_{distribution_type}.pkl')
            val_file = os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{client_id}_val_data_{distribution_type}.pkl')
            test_file = os.path.join(f'{self.client_data_dir}_{num_clients}', f'client_{client_id}_test_data_{distribution_type}.pkl')

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
