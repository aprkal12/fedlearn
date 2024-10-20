import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.Resnet_infer import Inference
import zstd

class ModelManager:
    def __init__(self):
        self.model = Inference()
        self.model.set_variable(client_id=4, non_iid_set=True, num_clients=5)
        self.model.set_epoch(3)
    
    def set_epoch(self, n):
        self.model.set_epoch(n)

    def load_params(self, params):
        self.model.load_parameter(params)
    
    def extract_params(self):
        return self.model.parameter_extract()
    
    def train_model(self):
        self.model.run()
    
    def compress_params(self, params):
        binary_data = pickle.dumps(params)
        return zstd.compress(binary_data)
    
    def decompress_params(self, comp_data):
        decomp_data = zstd.decompress(comp_data)
        return pickle.loads(decomp_data)