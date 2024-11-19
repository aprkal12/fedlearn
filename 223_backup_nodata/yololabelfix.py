import os
import json
from tqdm import tqdm

# 기존 COCO annotation 파일 경로 설정
train_annotation_file = 'instances_train2017.json'

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