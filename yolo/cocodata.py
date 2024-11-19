# import os
# import random
# import torch
# from torchvision.datasets import CocoDetection
# from torchvision import transforms
# from sklearn.model_selection import train_test_split
# from PIL import Image
# from tqdm import tqdm

# # COCO 데이터셋 경로 설정
# data_dir = './coco'
# train_dir = os.path.join(data_dir, 'train2017')
# val_dir = os.path.join(data_dir, 'val2017')
# train_annotation_file = os.path.join(data_dir, 'annotations/instances_train2017.json')
# val_annotation_file = os.path.join(data_dir, 'annotations/instances_val2017.json')

# # 변환 설정 및 데이터셋 불러오기
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = CocoDetection(root=train_dir, annFile=train_annotation_file, transform=transform)
# val_dataset = CocoDetection(root=val_dir, annFile=val_annotation_file, transform=transform)

# # 전체 데이터셋 결합 후 train, val, test로 분할
# combined_dataset = train_dataset + val_dataset
# train_indices, temp_indices = train_test_split(range(len(combined_dataset)), test_size=0.2, random_state=42)
# val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

# # 클라이언트 수 설정 및 경로 생성
# num_clients = 5
# output_dir = './coco_yolo_clients'
# os.makedirs(output_dir, exist_ok=True)

# # 각 클라이언트의 데이터를 YOLO 형식으로 저장하는 함수
# def save_yolo_format(client_id, indices, dataset, split):
#     img_save_dir = os.path.join(output_dir, f'client_{client_id}', split, 'images')
#     label_save_dir = os.path.join(output_dir, f'client_{client_id}', split, 'labels')
#     os.makedirs(img_save_dir, exist_ok=True)
#     os.makedirs(label_save_dir, exist_ok=True)

#     # 이미지마다 tqdm 적용
#     for idx in tqdm(indices, desc=f"Client {client_id} - {split}", unit="image"):
#         # 이미지와 주석 데이터 가져오기
#         image, annotations = dataset[idx]
#         if isinstance(image, torch.Tensor):
#             image = transforms.ToPILImage()(image)  # 텐서를 PIL 이미지로 변환

#         # 이미지 저장
#         img_id = f"{client_id}_{idx}"
#         img_path = os.path.join(img_save_dir, f"{img_id}.jpg")
#         image.save(img_path)

#         # 바운딩 박스를 YOLO 형식으로 변환하여 라벨 파일 저장
#         label_path = os.path.join(label_save_dir, f"{img_id}.txt")
#         img_width, img_height = image.size
#         with open(label_path, 'w') as f:
#             for ann in annotations:
#                 class_id = ann['category_id']
#                 x, y, width, height = ann['bbox']
#                 x_center = (x + width / 2) / img_width
#                 y_center = (y + height / 2) / img_height
#                 width /= img_width
#                 height /= img_height
#                 f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
# import os
# import yaml

# def create_client_data_yaml(client_id, output_dir="./coco_yolo_clients"):
#     # 클라이언트 데이터 경로 설정
#     train_path = os.path.join(output_dir, f"client_{client_id}", "train", "images")
#     val_path = os.path.join(output_dir, f"client_{client_id}", "val", "images")
    
#     # YOLO 모델에 맞는 클래스 이름 설정 (예시로 COCO 클래스 이름 사용)
#     class_names = [
#         "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
#         "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
#         "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
#         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
#         "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#         "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
#         "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#         "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
#         "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
#         "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
#         "toothbrush"
#     ]

#     # YAML 파일 내용 생성
#     data_yaml = {
#         "train": train_path,
#         "val": val_path,
#         "names": class_names
#     }

#     # YAML 파일 저장 경로
#     yaml_path = os.path.join(output_dir, f"client_{client_id}_data.yaml")
    
#     # YAML 파일 생성
#     with open(yaml_path, "w") as f:
#         yaml.dump(data_yaml, f)
    
#     print(f"YAML file created for client {client_id}: {yaml_path}")
#     return yaml_path

# if __name__=="__main__":
# # 예    시로 클라이언트 0의 YAML 파일 생성
#     for i in range(5):
#         yaml_path = create_client_data_yaml(client_id=i)




# # # 데이터셋을 클라이언트별로 나누어 YOLO 형식으로 저장
# # train_splits = [train_indices[i::num_clients] for i in range(num_clients)]
# # val_splits = [val_indices[i::num_clients] for i in range(num_clients)]
# # test_splits = [test_indices[i::num_clients] for i in range(num_clients)]

# # # 클라이언트별 데이터 저장
# # for client_id in tqdm(range(num_clients), desc="Processing Clients", unit="client"):
# #     print(f"\nProcessing Client {client_id}")
# #     save_yolo_format(client_id, train_splits[client_id], combined_dataset, 'train')
# #     save_yolo_format(client_id, val_splits[client_id], combined_dataset, 'val')
# #     save_yolo_format(client_id, test_splits[client_id], combined_dataset, 'test')

import os
import json
from tqdm import tqdm

# 기존 COCO annotation 파일 경로 설정
train_annotation_file = './coco/annotations/instances_train2017.json'

# category_id를 YOLO 형식의 연속적인 class_id로 매핑하기 위한 딕셔너리 생성
with open(train_annotation_file, 'r') as f:
    categories = json.load(f)['categories']
category_id_to_class_id = {cat['id']: i for i, cat in enumerate(categories)}

# 기존 라벨 디렉토리 설정
output_dir = './coco_yolo_clients'

# 각 클라이언트의 라벨을 YOLO 형식으로 업데이트하는 함수
def update_yolo_labels(client_id, split):
    label_dir = os.path.join(output_dir, f'client_{client_id}', split, 'labels')
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    # 각 라벨 파일을 읽고 class_id를 업데이트
    for label_file in tqdm(label_files, desc=f"Updating labels for Client {client_id} - {split}", unit="file"):
        label_path = os.path.join(label_dir, label_file)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # class_id 업데이트
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            original_class_id = int(parts[0])
            if original_class_id in category_id_to_class_id:
                new_class_id = category_id_to_class_id[original_class_id]
                updated_line = ' '.join([str(new_class_id)] + parts[1:]) + '\n'
                updated_lines.append(updated_line)
            else:
                print(f"Warning: Unrecognized category_id {original_class_id} in {label_file}")

        # 라벨 파일 업데이트
        with open(label_path, 'w') as f:
            f.writelines(updated_lines)

# 클라이언트별로 train, val, test 라벨 파일을 업데이트
for client_id in range(5):  # 예시로 5개 클라이언트 사용
    update_yolo_labels(client_id, 'train')
    update_yolo_labels(client_id, 'val')
    update_yolo_labels(client_id, 'test')

