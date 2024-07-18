# 192.168.0.191
import requests
import json

from requests.packages.urllib3.exceptions import InsecureRequestWarning
import torch
from Resnet_infer import Inference

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

data = {"전송" : "테스트", "숫자도 테스트" : 999}

url = "http://192.168.0.187:11110/initiation"

response = requests.post(url, json=data, verify=False)

count = 0
early = 0
model = Inference()
model.set_epoch(5)

while(True):
    count += 1
    receive_data = json.loads(response.text)
    print("서버에서 받아온 파라미터 타입 : ", type(receive_data))

    tensor_data = {key: torch.tensor(value) for key, value in receive_data.items()}
    print("텐서로 변환 후 : ", type(tensor_data))

    model.set_variable()
    model.load_parameter(tensor_data)
    model.run()

    update_param = model.parameter_extract()
    print("학습한 파라미터 추출")

    numpy_data = {key: model.tensor_to_numpy(value) for key, value in update_param.items()}
    print("파라미터 넘파이로 변환")

    json_data = json.dumps(numpy_data)
    print("json으로 변환")

    url = "http://192.168.0.187:11110/aggregation"
    response = requests.post(url, json=json_data, verify=False)

    if count > 30:
        print("테스트 종료")
        break
