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
    
    def set_uid(self, uid):
        return requests.post(f"{self.server_url}/client/name", data=uid, verify=False)

    def send_params(self, params):
        # name = utils.get_hostname()
        name = utils.get_name()
        
        b64_data = base64.b64encode(params).decode('utf-8') # base64 방식
        data = {
            'client_name': name,
            'params': b64_data # base64 방식
            # 'params': comp_data.hex() # 16진법 방식
        }

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
        # name = utils.get_hostname()
        name = utils.get_name()
        data = {'name' : name, 'signal' : signal}
        response = requests.post(f"{self.server_url}/parameter/signal", json=data)
        # print(response)
    
    def connect_socket(self, on_connect, on_disconnect, on_aggregated_params, on_train):
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('aggregated_params', on_aggregated_params)
        self.sio.on('training', on_train)
        self.sio.connect(self.server_url)

    def wait_socket(self):
        self.sio.wait()