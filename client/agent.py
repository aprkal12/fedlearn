import os
import sys
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_manager import ModelManager
from network_manager import NetworkManager
import utils

class ClientAgent:
    def __init__(self, server_url):
        self.model_manager = ModelManager()
        self.network_manager = NetworkManager(server_url)
        self.params = {}
        self.server_url = server_url
    
    def connect_to_server(self):
        self.network_manager.connect_socket(
            self.on_connect,
            self.on_disconnect,
            self.on_aggregated_params,
            self.on_train
        )

    def on_connect(self):
        print("서버 연결")
        utils.set_host()
        utils.set_id()
        name, ip = utils.get_host()
        uid = utils.get_id()
        # client_info = {"hostname": name, "id": uid}
        client_info = {"id": uid}
        response = self.network_manager.register_client(client_info)
        
        self.params = self.model_manager.decompress_params(response.content)

        response = self.network_manager.set_uid(uid)
        name = response.content.decode('utf-8')
        utils.set_name(name) # uid 기반으로 서버로부터 부여된 클라이언트 id (client1, client2 ...) 가져오기

        self.network_manager.post_params_signal("ready")
    
    def on_disconnect(self):
        print("서버 연결이 끊어졌습니다.")
        sys.exit(0)

    def on_aggregated_params(self):
        print("집계된 파라미터 수신")
        try:
            comp_data = self.network_manager.fetch_aggregated_params()
            if comp_data:
                self.params = self.model_manager.decompress_params(comp_data)
                self.network_manager.post_params_signal("ready")
        except Exception as e:
            self.network_manager.post_params_signal("error")
            print(f"Error during aggregated parameters processing: {e}")
    
    def on_train(self):
        print("train start")
        self.model_manager.load_params(self.params)
        self.network_manager.post_params_signal("training") # 서버로 상태 전송
        self.train()
        self.send()

    def train(self):
        self.model_manager.train_model()
    
    def send(self):
        updated_params = self.model_manager.extract_params()
        compressed_params = self.model_manager.compress_params(updated_params)
        self.network_manager.send_params(compressed_params)
        self.network_manager.post_params_signal("Finish")

    def start(self):
        self.connect_to_server()
        self.network_manager.wait_socket()