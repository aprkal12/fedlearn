import socketio
import requests

class NetworkManager:
    def __init__(self, server_url):
        self.server_url = server_url
        self.sio = socketio.Client()
    
    def register_client(self, client_info):
        return requests.post(f"{self.server_url}/client", json=client_info, verify=False)
    
    def send_params(self, params):
        response = requests.post(f"{self.server_url}/parameter", data=params)
        return response.text
    
    def fetch_aggregated_params(self):
        response = requests.get(f"{self.server_url}/parameter")
        if response.status_code == 200:
            return response.content
        else:
            print("아직 이번 라운드 파라미터가 집계되지 않았습니다.")
            return None
    
    def connect_socket(self, on_connect, on_disconnect, on_aggregated_params):
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('aggregated_params', on_aggregated_params)
        self.sio.connect(self.server_url)

    def wait_socket(self):
        self.sio.wait()