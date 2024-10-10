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
    
    def get_initial_params(self):
        return requests.get(f"{self.server_url}/client")
    
    def send_params(self, params):
        # name = utils.get_hostname()
        name = utils.get_name()
        
        print("파라미터 전송")
        response = requests.post(f"{self.server_url}/transmitter?name={name}", data=params)
        return response.text
    
    def fetch_aggregated_params(self):
        response = requests.get(f"{self.server_url}/transmitter")
        if response.status_code == 200:
            return response.content
        else:
            print("아직 이번 라운드 파라미터가 집계되지 않았습니다.")
            return None
    
    def post_params_signal(self, signal):
        # name = utils.get_hostname()
        name = utils.get_name()
        data = {'name' : name, 'signal' : signal}
        response = requests.post(f"{self.server_url}/transmitter/signal", json=data)
        # print(response)
    
    def connect_socket(self, on_connect, on_disconnect, on_aggregated_params, on_train, on_testcon):
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('aggregated_params', on_aggregated_params)
        self.sio.on('training', on_train)
        self.sio.on('test_connect', on_testcon)
        self.sio.connect(self.server_url)

    def wait_socket(self):
        self.sio.wait()
    
    def get_sid(self):
        return self.sio.sid
    
    def disconnect(self):
        self.sio.disconnect()