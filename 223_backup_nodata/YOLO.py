import json
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

import utils

def testmodel():
    model = YOLO("D:\\fed\\runs\\detect\\train2\\weights\\best.pt")
    # model = YOLO("yolov10m.pt")

    testimg = "D:\\fed\\coco_yolo_clients\\client_0\\test\\images\\0_1966.jpg"
    
    # # 총 파라미터 메모리 용량 계산 (MB 단위)
    # total_memory = 0
    # for param_tensor in model.model.state_dict().values():
    #     total_memory += param_tensor.numel() * param_tensor.element_size()  # 바이트 단위
    # total_memory_MB = total_memory / (1024 ** 2)  # MB로 변환
    # print(total_memory_MB)
    
    # model.model.to(torch.bfloat16) # float16으로 변환 -> YOLO에서 사용하는 방식 (메모리 추가 절약)
    # data = pickle.dumps(model.model.state_dict())
    # comp_data = zstd.compress(data)
    # total_memory_MB = sys.getsizeof(comp_data) / (1024 ** 2)
    # print(f"Total parameter memory: {total_memory_MB:.2f} MB")

    # # 모델의 파라미터를 float32와 bfloat16으로 변환하여 저장
    # parameters_float32 = {key: value.clone() for key, value in model.model.state_dict().items()}
    # parameters_bfloat16 = {key: value.to(torch.bfloat16).to(torch.float32) for key, value in parameters_float32.items()}

    # # 차이 유무 확인
    # difference_found = False

    # for key in parameters_float32.keys():
    #     if not torch.equal(parameters_float32[key], parameters_bfloat16[key]):
    #         print(f"Difference found in layer: {key}")
    #         difference_found = True

    # if not difference_found:
    #     print("No differences found between float32 and bfloat16 converted back to float32.")

    # 모델의 파라미터를 float32와 bfloat16으로 변환하여 저장
    parameters_float32 = {key: value.clone() for key, value in model.model.state_dict().items()}
    parameters_bfloat16 = {key: value.to(torch.bfloat16).to(torch.float32) for key, value in parameters_float32.items()}

    # 예시로 특정 레이어를 선택하여 값 출력
    layer_name = 'model.0.conv.weight'  # 예시로 확인할 레이어 이름
    if layer_name in parameters_float32:
        print(f"Layer: {layer_name}")
        print("Float32 values:")
        print(parameters_float32[layer_name])
        
        print("\nConverted bfloat16 -> float32 values:")
        print(parameters_bfloat16[layer_name])

        # 두 값의 차이 출력
        diff = (parameters_float32[layer_name] - parameters_bfloat16[layer_name]).abs()
        print("\nDifference:")
        print(diff)
    else:
        print(f"Layer '{layer_name}' not found in model parameters.")



    # # # 1. 텐서를 넘파이 배열로 변환 (먼저 CUDA 텐서를 CPU로 복사)
    # parameters = model.model.state_dict()  # 예시로 파라미터 추출
    # # print(parameters.keys())
    # data = pickle.dumps(parameters)
    # # parameters_as_numpy = {k: v.cpu().numpy() for k, v in parameters.items()}  # .cpu() 추가

    # # # 2. 넘파이 배열을 JSON으로 직렬화하려면 넘파이 배열을 리스트로 변환해야 함
    # # parameters_as_list = {k: v.tolist() for k, v in parameters_as_numpy.items()}

    # # # 3. JSON 형식으로 직렬화
    # # json_data = json.dumps(parameters_as_list)

    # print(f"파라미터 크기 : {utils.bytes_to_mb(len(data)):.2f} MB" )

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
    # model.float() # 다시 float32로 변환
    # yaml_path = "D:\\fed\\coco_yolo_clients\\client_0_data.yaml"

    # # 평가 수행 (val 메서드를 사용하여 정확도 계산)
    # results = model.val(data=yaml_path, split="test")  # yaml 파일을 통해 데이터 로드

    # # 정확도 출력
    # accuracy = results.box.map * 100  # mAP@0.5 (Mean Average Precision at 0.5 IOU)
    # print(f"Model Accuracy (mAP@0.5): {accuracy:.2f}%")

if __name__=="__main__":
    testmodel()
    # 미리 학습된 가중치 없이 YOLO 모델 초기화
    # model = YOLO("yolov10s.yaml")

    # yaml_path = "D:\\fed\\coco_yolo_clients\\client_0_data.yaml"

    # # 클라이언트 학습 실행 (예: Client 0에서 학습)
    # model.train(data=yaml_path, epochs=5)


    # # 모델 파라미터 추출
    # model_params = model.model.state_dict()

    # # 모델 파라미터 로드 예시 (다른 클라이언트로 전송받은 모델 사용 시)
    # model.model.load_state_dict(model_params)
