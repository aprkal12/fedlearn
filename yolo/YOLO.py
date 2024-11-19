import sys
import zstd
from matplotlib import pyplot as plt
from ultralytics import YOLO
from torch.utils.data import DataLoader
import torch
import pickle
import os
from tqdm import tqdm
from PIL import Image
import glob
from torchvision.transforms import functional as F
from torchsummary import summary

def testmodel():
    model = YOLO("runs\\detect\\train35\\weights\\best.pt")
    # model = YOLO("yolov10m.pt")

    testimg = "D:\\fedlearn\\coco_yolo_clients\\client_0\\test\\images\\0_102.jpg"
    
    # # 총 파라미터 메모리 용량 계산 (MB 단위)
    # total_memory = 0
    # for param_tensor in model.model.state_dict().values():
    #     total_memory += param_tensor.numel() * param_tensor.element_size()  # 바이트 단위
    # total_memory_MB = total_memory / (1024 ** 2)  # MB로 변환
    
    model.half() # float16으로 변환 -> YOLO에서 사용하는 방식 (메모리 추가 절약)
    # data = pickle.dumps(model.model.state_dict())
    # comp_data = zstd.compress(data)
    # total_memory_MB = sys.getsizeof(comp_data) / (1024 ** 2)
    # print(f"Total parameter memory: {total_memory_MB:.2f} MB")

    # # 이미지 예측 수행
    # # Perform object detection on an image
    # results = model(testimg)

    # # Display the results
    # results[0].show()

    # # 예측 결과 시각화
    # for result in results:
    #     img_with_boxes = result.plot()  # 예측된 바운딩 박스가 표시된 이미지를 반환
    #     plt.imshow(img_with_boxes)
    #     plt.axis('off')
    #     plt.show()
    # 평가를 위한 yaml 파일 경로
    model.float() # 다시 float32로 변환
    yaml_path = "D:\\fedlearn\\coco_yolo_clients\\client_0_data.yaml"

    # 평가 수행 (val 메서드를 사용하여 정확도 계산)
    results = model.val(data=yaml_path, split="test")  # yaml 파일을 통해 데이터 로드

    # 정확도 출력
    accuracy = results.box.map * 100  # mAP@0.5 (Mean Average Precision at 0.5 IOU)
    print(f"Model Accuracy (mAP@0.5): {accuracy:.2f}%")

if __name__=="__main__":
    testmodel()
    # # 미리 학습된 가중치 없이 YOLO 모델 초기화
    # model = YOLO("yolov10l.yaml")

    # yaml_path = "D:\\fedlearn\\coco_yolo_clients\\client_0_data.yaml"

    # # 클라이언트 학습 실행 (예: Client 0에서 학습)
    # model.train(data=yaml_path, epochs=5)


    # # 모델 파라미터 추출
    # model_params = model.model.state_dict()

    # # 모델 파라미터 로드 예시 (다른 클라이언트로 전송받은 모델 사용 시)
    # model.model.load_state_dict(model_params)
