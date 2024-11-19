import os
import pickle
import sys

import torch
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import zstd

class YOLOModelManager:
    def __init__(self):
        self.model = YOLO("yolov10s.yaml")
        self.yaml_path = "D:\\fed\\coco_yolo_clients\\client_0_data.yaml"

    def load_params(self, params):
        self.model.model.load_state_dict(params)
        self.model.model.float()
    
    def extract_params(self):
        self.model.model.to(torch.bfloat16)
        return self.model.model.state_dict()
    
    def train_model(self):
        self.model.train(data=self.yaml_path, epochs=5, batch=8, imgsz=640)
    
    def compress_params(self, params):
        binary_data = pickle.dumps(params)
        return zstd.compress(binary_data)
    
    def decompress_params(self, comp_data):
        decomp_data = zstd.decompress(comp_data)
        return pickle.loads(decomp_data)