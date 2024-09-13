import os
import pickle
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from collections import defaultdict

def check_data_distribution(num_clients):
    """
    각 클라이언트의 데이터셋에서 클래스별 분포 및 총량을 확인하는 함수.
    
    :param num_clients: 클라이언트 수
    """
    data_dir = './cifar100_non_iid'
    
    for client_id in range(num_clients):
        print(f"\nClient {client_id} data distribution:")

        # 각 데이터셋(train, validation, test) 분포 및 데이터 개수 확인
        for dataset_type in ['train', 'val', 'test']:
            with open(os.path.join(data_dir, f'client_{client_id}_{dataset_type}_data.pkl'), 'rb') as f:
                client_data = pickle.load(f)

            # 클래스별 샘플 개수 계산
            class_counts = defaultdict(int)
            for _, label in client_data:
                class_counts[label] += 1

            # 클래스별 샘플 개수 요약해서 출력
            print(f"  {dataset_type.capitalize()} set total samples: {sum(class_counts.values())}")
            print(f"  Class-wise sample counts: {list(class_counts.values())}")

def split_cifar100_non_iid(num_clients, data_size=1.0):
    """
    CIFAR-100 데이터셋을 8:1:1 비율로 나누고, Non-IID 방식으로 각 클라이언트에 데이터를 분배합니다.
    
    :param num_clients: 클라이언트 수
    :param data_size: 데이터 크기 비율 (0~1 사이의 값)
    """
    data_dir = './cifar100_non_iid'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

     # CIFAR-100 데이터 다운로드 및 준비 (train dataset만 사용)
    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transforms.ToTensor(), download=True)

    total_train_size = int(len(train_dataset) * data_size)
    
    # 각 클래스별로 데이터셋 분리
    class_data = {i: [] for i in range(100)}  # CIFAR-100은 100개의 클래스
    for idx in range(total_train_size):
        _, label = train_dataset[idx]
        class_data[label].append(idx)

    # 클라이언트별 데이터셋 초기화
    client_train_data = {i: [] for i in range(num_clients)}
    client_val_data = {i: [] for i in range(num_clients)}
    client_test_data = {i: [] for i in range(num_clients)}

    # 각 클래스별로 데이터를 8:1:1 비율로 정확히 분배
    for class_id, class_list in class_data.items():
        np.random.shuffle(class_list)
        class_size = len(class_list)

        # 전체 데이터를 8:1:1 비율로 나눕니다 (train, validation, test)
        num_train = int(0.8 * class_size)
        num_val = int(0.1 * class_size)
        num_test = class_size - num_train - num_val

        train_data = class_list[:num_train]
        val_data = class_list[num_train:num_train + num_val]
        test_data = class_list[num_train + num_val:]

        # Non-IID로 데이터를 분배하기 위해 각 클라이언트에 무작위 비율로 데이터를 할당
        train_proportions = np.random.dirichlet(np.ones(num_clients))
        val_proportions = np.random.dirichlet(np.ones(num_clients))
        test_proportions = np.random.dirichlet(np.ones(num_clients))

        train_start_idx = 0
        val_start_idx = 0
        test_start_idx = 0
        
        for i in range(num_clients):
            # 각 클라이언트별로 할당된 비율에 맞춰 데이터를 분배
            train_samples = int(train_proportions[i] * num_train)
            val_samples = int(val_proportions[i] * num_val)
            test_samples = int(test_proportions[i] * num_test)

            # 데이터 할당
            client_train_data[i].extend(train_data[train_start_idx:train_start_idx + train_samples])
            client_val_data[i].extend(val_data[val_start_idx:val_start_idx + val_samples])
            client_test_data[i].extend(test_data[test_start_idx:test_start_idx + test_samples])

            # 인덱스 갱신
            train_start_idx += train_samples
            val_start_idx += val_samples
            test_start_idx += test_samples

    # 데이터를 파일로 저장
    for i in range(num_clients):
        with open(os.path.join(data_dir, f'client_{i}_train_data.pkl'), 'wb') as f:
            pickle.dump(Subset(train_dataset, client_train_data[i]), f)
        with open(os.path.join(data_dir, f'client_{i}_val_data.pkl'), 'wb') as f:
            pickle.dump(Subset(train_dataset, client_val_data[i]), f)
        with open(os.path.join(data_dir, f'client_{i}_test_data.pkl'), 'wb') as f:
            pickle.dump(Subset(train_dataset, client_test_data[i]), f)  # 여기도 train dataset 사용

    print(f"Client data saved to {data_dir}")

# split_cifar100_non_iid(num_clients=5, data_size=1.0)
check_data_distribution(num_clients=5)

