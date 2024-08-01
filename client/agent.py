from model_manager import ModelManager
from network_manager import NetworkManager
import utils

class ClientAgent:
    def __init__(self, server_url):
        self.model_manager = ModelManager()
        self.network_manager = NetworkManager(server_url)
        self.server_url = server_url
    
    def connect_to_server(self):
        self.network_manager.connect_socket(
            self.on_connect,
            self.on_disconnect,
            self.on_aggregated_params
        )

    def on_connect(self):
        print("서버 연결")
        name, ip = utils.set_host()
        client_info = {"hostname": name, "ip": ip}
        response = self.network_manager.register_client(client_info)
        
        params = self.model_manager.decompress_params(response.content)
        self.model_manager.load_params(params)
        self.train_and_send()
    
    def on_disconnect(self):
        print("서버 연결이 끊어졌습니다.")

    def on_aggregated_params(self):
        print("집계된 파라미터 수신")
        try:
            comp_data = self.network_manager.fetch_aggregated_params()
            if comp_data:
                params = self.model_manager.decompress_params(comp_data)
                self.model_manager.load_params(params)
                self.train_and_send()
        except Exception as e:
            print(f"Error during aggregated parameters processing: {e}")

    def train_and_send(self):
        self.model_manager.train_model()
        updated_params = self.model_manager.extract_params()
        compressed_params = self.model_manager.compress_params(updated_params)
        self.network_manager.send_params(compressed_params)

    def start(self):
        self.connect_to_server()
        self.network_manager.wait_socket()