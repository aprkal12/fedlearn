
import json
from flask import Flask, request, send_file
import torch
from Resnet_infer import Inference
from modeltest import base_model
import wandb

app = Flask(__name__)

wandb.init(
    project="Federated Learning",
    entity="aprkal12",
    config={
        "learning_rate": 0.001,
        "architecture": "Resnet18",
        "dataset": "CIFAR-10",
    }
)
wandb.run.name = "Resnet18_CIFAR-10_B=50%_E=5"

# model = base_model()
count = 0
c_num = 0 # 접속한 클라이언트 수
post_num = 0 # 파라미터를 보낸 클라이언트 수
logging = True
datas = None
val_loss = None
val_metric = None
model = Inference()


# @app.route('/send_file', methods=['POST'])
# def send_files():
#     return send_file('D:\Fedtest\Resnet_setdata.py')

@app.route('/aggregation', methods=['POST'])
def send_param():
    global count
    global c_num
    global post_num
    global logging
    global val_loss
    global val_metric

    tensorlist = []
    avg_weights = {}
    
    data = request.json
    post_num += 1

    if not logging: # 집계된 파라미터가 있을 때
        print("로그 기록")
        val_loss, val_metric = model.get_accuracy(model.model)
        wandb.log({"val loss" : val_loss, "accuracy" : val_metric})
        logging = True

    print('-'*10)
    print("파라미터 수신 대기중 ", post_num," / ",  c_num)

    received_data = json.loads(data)
    print("클라이언트로부터 받은 파라미터 : ", type(received_data))
    tensor_data = {key: torch.tensor(value) for key, value in received_data.items()}
    print("텐서로 변환 후 : ", type(tensor_data))

    tensorlist.append(tensor_data)
    print("가중치 리스트에 저장")
    while(True):
        if post_num == c_num:
            print('-'*10)
            print("모든 파라미터 수신완료")
            count += 1

            # Long dtype을 부동 소수점으로 변환
            tensorlist_float = [{key: value.float() for key, value in client_weights.items()} for client_weights in tensorlist]

            # 부동 소수점으로 변환한 데이터를 사용하여 평균 계산
            
            for key in tensorlist_float[0].keys():
                avg_weights[key] = torch.stack([client_weights[key] for client_weights in tensorlist_float], dim=0).mean(dim=0)
            model.load_parameter(avg_weights)            
            
            # 전역 모델에 평균 가중치 적용
            # model.load_parameter(avg_weights)
            # model.run()
            
            # 새로운 파라미터 전송을 위해 변수 초기화
            tensorlist.clear()
            count = 0
            post_num = 0

            numpy_data = {key: model.tensor_to_numpy(value) for key, value in avg_weights.items()}
            print("평균 파라미터 numpy로 변환 : ", type(numpy_data)) # 추출한 모델 파라미터를 numpy로 변환한 데이터

            jsondata = json.dumps(numpy_data)
            print("평균 파라미터 json으로 변환 : ", type(jsondata)) # numpy로 변환한 데이터를 json으로 변환한 데이터

            # print(f"\nRound {count}")
            # model.set_variable()
            # model.load_parameter(tensor_data)
            # model.run()

            # datas = model.parameter_extract() # 모델에서 파라미터 추출
            # print("추출된 파라미터 타입 : ", type(datas))

            # numpy_data = {key: model.tensor_to_numpy(value) for key, value in datas.items()}
            # print("numpy로 변환 후 : ", type(numpy_data)) # 추출한 모델 파라미터를 numpy로 변환한 데이터

            # jsondata = json.dumps(numpy_data)
            # print("json으로 변환 후 : ", type(jsondata)) # numpy로 변환한 데이터를 json으로 변환한 데이터
            print("파라미터 전송")
            logging = False
            return jsondata

@app.route('/initiation', methods=['POST'])
def send_data():
    global c_num
    global datas
    data = request.json  # 클라이언트로부터 전송된 JSON 데이터 받기
    # 클라이언트에게 응답 보내기
    response_data = {"message": "데이터 수신 성공"}
    c_num += 1

    # 클라이언트에서 송신한 모델 파라미터들을 평균내서 글로벌 모델에 적용시키면 됨
    
    print("추출된 파라미터 타입 : ", type(datas))

    numpy_data = {key: model.tensor_to_numpy(value) for key, value in datas.items()}
    print("numpy로 변환 : ", type(numpy_data)) # 추출한 모델 파라미터를 numpy로 변환한 데이터

    jsondata = json.dumps(numpy_data)
    print("json으로 변환 : ", type(jsondata)) # numpy로 변환한 데이터를 json으로 변환한 데이터

    return jsondata
    # return response_data, 200

if __name__ == '__main__':
    model.set_variable()
    wandb.watch(model.model)

    model.set_epoch(1)
    model.run()

    val_loss, val_metric = model.get_accuracy(model.model)
    wandb.log({"val loss" : val_loss, "accuracy" : val_metric})
    
    # wandb.log({"val loss" : val_loss, "accuracy" : val_metric})

    datas = model.parameter_extract() # 모델에서 파라미터 추출
    print("초기 파라미터 추출 완료")
    app.run('0.0.0.0', port=11110)