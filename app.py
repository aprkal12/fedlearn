import json
from flask import Flask, request, send_file, render_template
import requests
import torch
from Resnet_infer import Inference
import wandb
import zstd
import pickle

app = Flask(__name__)

##################################
count = 0
c_num = 0 # 접속한 클라이언트 수
post_num = 0 # 파라미터를 보낸 클라이언트 수
logging = True
datas = None
val_loss = None
val_metric = None
client_list = {}
model = Inference()
###################################


@app.route('/parameter', methods=['POST'])
def send_param():
    global count
    global c_num
    global post_num
    global logging
    global val_loss
    global val_metric

    tensorlist = []
    avg_weights = {}
    
    # data = request.json
    # data = request.data

    comp_data = request.data
    decomp_data = zstd.decompress(comp_data)
    model.load_parameter(pickle.loads(decomp_data))
    print("받은 파라미터 적용 완료")
    model.get_accuracy(model.model)
    print("성공인듯?")
    return "성공인듯?"

@app.route('/client', methods=['POST'])
def send_data():
    global client_list

    data = request.json  # 클라이언트로부터 전송된 JSON 데이터 받기
    client_name = data['hostname']
    client_ip = data['ip']
    client_list[client_name] = client_ip

    params = model.parameter_extract()
    binary_data = pickle.dumps(params)
    comp_data = zstd.compress(binary_data)

    return comp_data

@app.route('/')
def mainpage():
    return render_template('index.html', clients=client_list)

if __name__ == '__main__':
    model.set_variable()
    # wandb.watch(model.model)
    model.set_epoch(1)
    model.run()

    datas = model.parameter_extract() # 모델에서 파라미터 추출
    print("초기 파라미터 추출 완료")
    app.run('0.0.0.0', port=11110)
