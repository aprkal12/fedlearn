# 192.168.0.191
import requests
import json
import msgpack
import zstd
import pickle
import socket
import socketio
import time
# from websocket import WebSocketApp

from urllib3.exceptions import InsecureRequestWarning
import torch
from Resnet_infer import Inference

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

server_url = 'http://192.168.0.209:11110'
# ws_url = 'ws://192.168.0.209:11110/socket.io/?EIO=4&transport=websocket'
sio = socketio.Client()

model = Inference()
model.set_variable()
model.set_epoch(5)

def set_host():
    hostname = socket.gethostname()
    hostip = socket.gethostbyname(hostname)
    return hostname, hostip

def send_params(params):
    global server_url
    binary_data = pickle.dumps(params)
    comp_data = zstd.compress(binary_data)
    response = requests.post(f"{server_url}/parameter", data = comp_data)
    print("파라미터 전송")
    print(response.text)

def fetch_aggregated_params():
    global server_url
    response = requests.get(f"{server_url}/parameter")
    print("집계된 파라미터 수신")
    if response.status_code == 200: # get 성공
        comp_data = response.content
        decomp_data = zstd.decompress(comp_data)
        aggregated_params = pickle.loads(decomp_data)
        return aggregated_params
    else:
        print("아직 이번 라운드 파라미터가 집계되지 않았습니다.")
        return None

def train_model():
    global model
    model.run()

@sio.event
def connect():
    global server_url
    global model

    print("서버 연결")

    url = f"{server_url}/client"
    name, ip = set_host()
    client_info = {"hostname":name, "ip":ip}
    response = requests.post(url, json=client_info, verify=False)

    comp_data = response.content
    decomp_data = zstd.decompress(comp_data)
    model.load_parameter(pickle.loads(decomp_data))
    train_model()
    updated_params = model.parameter_extract()
    send_params(updated_params)

    print("클라이언트 등록 절차 완료")

@sio.on('aggregated_params')
def aggregated_params():
    global model
    print("집계완료 신호 수신")
    try:
        aggregated_params = fetch_aggregated_params()
        if aggregated_params is not None:
            model.load_parameter(aggregated_params)
            train_model()
            updated_params = model.parameter_extract()
            send_params(updated_params)
    except Exception as e:
        print(f"Error processing aggregated parameters: {e}")

# @sio.on('test')
# def testfunc():
    

@sio.event
def disconnect():
    global server_url
    print("### closed ###")
    sio.connect(server_url)  # 즉시 재연결 시도

@sio.event
def connect_error(data):
    print(f"Connection failed: {data}")

def main():
    global server_url
    sio.connect(server_url)

    sio.wait()