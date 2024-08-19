import base64
import json
import pickle
import socketio
import requests
import zstd
import utils

class NetworkManager:
    def __init__(self, server_url):
        self.server_url = server_url
        self.sio = socketio.Client()
    
    def register_client(self, client_info):
        return requests.post(f"{self.server_url}/client", json=client_info, verify=False)
    
    # 파일 크기 mb 변환
    def bytes_to_mb(self, size_in_bytes):
        return size_in_bytes / (1024 * 1024)

    def send_params(self, params):
        name, ip = utils.get_host()
        
        # comp_data = zstd.compress(pickle.dumps(params))
        # size_existing_method = len(comp_data)
        # print(f"기존 방식 데이터 크기: {self.bytes_to_mb(size_existing_method):.6f} MB")

        b64_data = base64.b64encode(params).decode('utf-8') # base64 방식
        data = {
            'client_name': name,
            'params': b64_data # base64 방식
            # 'params': comp_data.hex() # 16진법 방식
        }

        # json_str = json.dumps(data)
        # size_new_method = len(json_str.encode('utf-8'))
        # print(f"새로운 방식 데이터 크기: {self.bytes_to_mb(size_new_method):.6f} MB")
        print("파라미터 전송")
        response = requests.post(f"{self.server_url}/parameter", json=data)
        return response.text
    
    def fetch_aggregated_params(self):
        response = requests.get(f"{self.server_url}/parameter")
        if response.status_code == 200:
            return response.content
        else:
            print("아직 이번 라운드 파라미터가 집계되지 않았습니다.")
            return None
    
    def post_params_signal(self, signal):
        name = utils.get_hostname()
        data = {'name' : name, 'signal' : signal}
        response = requests.post(f"{self.server_url}/parameter/signal", json=data)
        print(response)
    
    def connect_socket(self, on_connect, on_disconnect, on_aggregated_params, on_train):
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('aggregated_params', on_aggregated_params)
        self.sio.on('train', on_train)
        self.sio.connect(self.server_url)

    def wait_socket(self):
        self.sio.wait()