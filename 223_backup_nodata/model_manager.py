import os
import pickle
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Resnet_infer import Inference
import zstd

class ModelManager:
    def __init__(self):
        self.model = Inference()
        self.model.set_variable(client_id=0, non_iid_set=True, num_clients=2)
        self.model.set_epoch(2)
    
    def set_epoch(self, n):
        self.model.set_epoch(n)

    def load_params(self, params):
        self.model.load_parameter(params)
        self.model.model.float()
    
    def extract_params(self):
        self.model.model.to(torch.bfloat16)
        return self.model.parameter_extract()
    
    def train_model(self):
        self.model.run()
    
    def compress_params(self, params):
        binary_data = pickle.dumps(params)
        return zstd.compress(binary_data)
    
    def decompress_params(self, comp_data):
        decomp_data = zstd.decompress(comp_data)
        return pickle.loads(decomp_data)