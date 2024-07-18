# from gevent import monkey
# monkey.patch_all()

import json
import threading
from flask import Flask, request, send_file, render_template
from flask_socketio import SocketIO, emit
import requests
import torch
from Resnet_infer import Inference
import wandb
import zstd
import pickle
import logging

app = Flask(__name__)
socketio = SocketIO(app)

# logging.basicConfig(filename='server.log', level=logging.DEBUG)
# wandb.init(
#     project="Federated_Learning",
#     entity="aprkal12",
#     config={
#         "learning_rate": 0.001,
#         "architecture": "Resnet18",
#         "dataset": "CIFAR-10",
#     }
# )
# wandb.run.name = "Resnet18_CIFAR-10_B=100%_E=1"

count = 0
c_num = 0 # 접속한 클라이언트 수
post_num = 0 # 파라미터를 보낸 클라이언트 수
logging = True
datas = None
val_loss = None
val_metric = None

client_list = {}

model = Inference()

parameters = []
parameter_lock = threading.Lock()
expected_clients = 2
avg_weights = None

# @app.route('/send_file', methods=['POST'])
# def send_files():
#     return send_file('D:\Fedtest\Resnet_setdata.py')

@app.route('/parameter', methods=['POST', 'GET'])
def parameter_module():
    global post_num
    global parameters
    global avg_weights

    if request.method == 'POST': # 클라이언트로부터 파라미터 수신
        comp_data = request.data
        decomp_data = zstd.decompress(comp_data)
        client_params = pickle.loads(decomp_data)
        print("파라미터 수신 확인")
        with parameter_lock:
            parameters.append(client_params)
            post_num += 1
            print("post_num : ", post_num)

        return "서버 : 파라미터 전송 완료"
    
    elif request.method == 'GET':
        if avg_weights is None:
            return "이번 라운드의 평균 파라미터가 아직 집계되지 않았습니다.", 400
        binary_data = pickle.dumps(avg_weights)
        comp_data = zstd.compress(binary_data)
        return comp_data
    
@app.route('/aggregate', methods=['POST'])
def aggregate():
    global post_num
    global parameters
    global expected_clients
    global avg_weights
    
    print("post_num : ", post_num)
    print("expected_clients : ", expected_clients)
    with parameter_lock:
        if post_num == expected_clients:
            avg_weights = {}
            tensorlist_float = [{key: value.float() for key, value in client_weights.items()} for client_weights in parameters]
            for key in tensorlist_float[0].keys():
                avg_weights[key] = torch.stack([client_weights[key] for client_weights in tensorlist_float], dim=0).mean(dim=0)
            
            model.load_parameter(avg_weights)
            print("모든 파라미터 수신 완료 및 평균 파라미터 계산 완료")
            val_loss, val_metric = model.get_accuracy(model.model)
            print("val loss: %.6f, accuracy: %.2f %%" %(val_loss, 100*val_metric))
            parameters.clear()
            post_num = 0

            socketio.emit('aggregated_params')
            socketio.sleep(0)
            print("event emitted")
            return "집계 완료"
        else:
            print("집계 조건 충족 안됨")
    return "아직 모든 파라미터가 수신되지 않았습니다."

@app.route('/client', methods=['POST'])
def send_data():
    global client_list

    data = request.json  # 클라이언트로부터 전송된 JSON 데이터 받기
    client_name = data['hostname']
    client_ip = data['ip']
    client_list[client_name] = client_ip
    print("클라이언트 접속 확인")
    print(client_list)

    params = model.parameter_extract()
    binary_data = pickle.dumps(params)
    comp_data = zstd.compress(binary_data)

    return comp_data

@app.route('/delete', methods=['DELETE'])
def delete_clients():
    global client_list
    # client_list.clear()
    return "정보 삭제 완료"

@app.route('/trigger', methods=['GET'])
def trigger():
    target_info = request.json
    target_client = None

    for client_name, client_ip in client_list.items():
        if client_name == target_info['hostname']:
            target_client = client_name
            break
    
    if target_client is None:
        return "해당 클라이언트가 존재하지 않습니다.", 400
    else:
        print(f"sending trigger to {target_client}")
        socketio.emit('trigger', room = target_client)
        return "트리거 전송 완료", 200

@app.route('/')
def mainpage():
    return render_template('index.html', clients=client_list)

# def hello():
#     wandb_pjname = "Federated_Learning"
#     wandb_entity = "aprkal12"
#     runs = wandb.Api().runs(f"{wandb_entity}/{wandb_pjname}")
#     run = next(runs)
#     report_url = run.url

#     return render_template('index.html', report_url=report_url)


if __name__ == '__main__':
    model.set_variable()
    # wandb.watch(model.model)
    model.set_epoch(1)
    model.run()
    print("모델 학습 완료")
    datas = model.parameter_extract() # 모델에서 파라미터 추출
    print("초기 파라미터 추출 완료")
    print("서버 실행 중")
    # app.run('0.0.0.0', port=11110)
    socketio.run(app, host='0.0.0.0', port=11110)





# 파일 형식의 연합학습
    # params = model.parameter_extract()
    # torch.save(params, "models\\send_from_server.pt")

    # print("클라이언트한테 파일 전송")

    # return send_file("models\\send_from_server.pt", as_attachment=True) # 파일명도 보내기 위한 코드
    # ==================================================

    # 클라이언트에게 응답 보내기
    # response_data = {"message": "데이터 수신 성공"}
    # c_num += 1

    # # 클라이언트에서 송신한 모델 파라미터들을 평균내서 글로벌 모델에 적용시키면 됨
    
    # print("추출된 파라미터 타입 : ", type(datas))

    # numpy_data = {key: model.tensor_to_numpy(value) for key, value in datas.items()}
    # print("numpy로 변환 : ", type(numpy_data)) # 추출한 모델 파라미터를 numpy로 변환한 데이터

    # jsondata = json.dumps(numpy_data)
    # print("json으로 변환 : ", type(jsondata)) # numpy로 변환한 데이터를 json으로 변환한 데이터

    # # binary_data = msgpack.packb(jsondata)

    # return jsondata
    # return response_data, 200











    # 파일 방식의 파라미터 전송 ==================
    # if 'file' not in request.files:
    #     return 'No file part'

    # file = request.files['file']

    # if file.filename == '':
    #     return 'No selected file'

    # # 클라이언트가 보낸 파일을 저장할 경로
    # file.save(f'models\\{file.filename}')

    # model.load_parameter(torch.load(f'models\\{file.filename}'))

    # tensorlist.append(torch.load(f'models\\{file.filename}'))
    # print("파라미터 전송")
    # return "성공인듯?"
    # 파일 방식의 파라미터 전송 =======================

    # post_num += 1

    # if not logging: # 집계된 파라미터가 있을 때
    #     print("로그 기록")
    #     val_loss, val_metric = model.get_accuracy(model.model)
    #     print("val loss: %.6f, accuracy: %.2f %%" %(val_loss, 100*val_metric))
            
    #     # wandb.log({"val loss" : val_loss, "accuracy" : val_metric})
    #     logging = True

    # print('-'*10)
    # print("파라미터 수신 대기중 ", post_num," / ",  c_num)

    # received_data = json.loads(data)
    # # received_data = json.loads(msgpack.unpackb(data, raw=False))
    # print("클라이언트로부터 받은 파라미터 : ", type(received_data))
    # tensor_data = {key: torch.tensor(value) for key, value in received_data.items()}
    # print("텐서로 변환 후 : ", type(tensor_data))

    # tensorlist.append(tensor_data)
    # print("가중치 리스트에 저장")
    # while(True):
    #     if post_num == c_num:
    #         print('-'*10)
    #         print("모든 파라미터 수신완료")
    #         count += 1

    #         # Long dtype을 부동 소수점으로 변환
    #         tensorlist_float = [{key: value.float() for key, value in client_weights.items()} for client_weights in tensorlist]

    #         # 부동 소수점으로 변환한 데이터를 사용하여 평균 계산
            
    #         for key in tensorlist_float[0].keys():
    #             avg_weights[key] = torch.stack([client_weights[key] for client_weights in tensorlist_float], dim=0).mean(dim=0)
    #         model.load_parameter(avg_weights)            
            
    #         # 전역 모델에 평균 가중치 적용
    #         # model.load_parameter(avg_weights)
    #         # val_loss, val_metric = model.get_accuracy(model.model)
            

            
    #         # 새로운 파라미터 전송을 위해 변수 초기화
    #         tensorlist.clear()
    #         count = 0
    #         post_num = 0

    #         numpy_data = {key: model.tensor_to_numpy(value) for key, value in avg_weights.items()}
    #         print("평균 파라미터 numpy로 변환 : ", type(numpy_data)) # 추출한 모델 파라미터를 numpy로 변환한 데이터

    #         jsondata = json.dumps(numpy_data)
    #         print("평균 파라미터 json으로 변환 : ", type(jsondata)) # numpy로 변환한 데이터를 json으로 변환한 데이터

    #         binary_data = msgpack.packb(jsondata) # binary로 변환

    #         # print(f"\nRound {count}")
    #         # model.set_variable()
    #         # model.load_parameter(tensor_data)
    #         # model.run()

    #         # datas = model.parameter_extract() # 모델에서 파라미터 추출
    #         # print("추출된 파라미터 타입 : ", type(datas))

    #         # numpy_data = {key: model.tensor_to_numpy(value) for key, value in datas.items()}
    #         # print("numpy로 변환 후 : ", type(numpy_data)) # 추출한 모델 파라미터를 numpy로 변환한 데이터

    #         # jsondata = json.dumps(numpy_data)
    #         # print("json으로 변환 후 : ", type(jsondata)) # numpy로 변환한 데이터를 json으로 변환한 데이터
    #         print("파라미터 전송")
    #         logging = False
    #         return jsondata
    #         # return send_file(model.get_params_file)