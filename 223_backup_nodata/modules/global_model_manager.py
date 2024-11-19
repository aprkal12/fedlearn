import datetime
from flask import current_app, request, g
from flask_socketio import emit, SocketIO
from ultralytics import YOLO
from . import aggregate_bp
import threading
import torch
from Resnet_infer import Inference
import global_vars as gv
import wandb

from modules.client_manager import send_reload_signal, send_aggregated_signal

parameter_lock = threading.Lock()
expected_clients = len(gv.client_list)

@aggregate_bp.route('/aggregate', methods=['POST'])
def round_manager():
    msg = aggregate_parameters()
    if msg == "aggregated": 
        # gv.round_num += 1
        send_reload_signal()
        global_model_update()

        print("round %d complete" % gv.round_num)
        print("="*10)
        print()
        print("next round setting...")
        next_round_set()
        # gv.socketio.emit('aggregated_params')
        send_aggregated_signal()
    return msg


def aggregate_parameters():
    global expected_clients
    expected_clients = len(gv.client_list)
    with parameter_lock:
        if gv.post_num == expected_clients:
            for status in gv.client_status.values():
                if status != "finish":
                    # return "이전 라운드의 학습이 완료되지 않았습니다."
                    print("The previous round's training is not complete.")
                    return "The previous round's training is not complete."
            print("parameter aggregation start")
            gv.avg_weights = {}
            # tensorlist_float = [{key: value.float() for key, value in client_weights.items()} for client_weights in gv.parameters]
            tensorlist_float = [{key: value.float() for key, value in client_weights.items()} for client_weights in (client_data for client_data in gv.parameters.values())]

            for key in tensorlist_float[0].keys():
                gv.avg_weights[key] = torch.stack([client_weights[key] for client_weights in tensorlist_float], dim=0).mean(dim=0)
            print("avg_weights aggregated")

            
            return "aggregated"
        else:
            print("집계 조건 충족 안됨")
    return "All parameters have not been received yet."

# def notify_clients():
#     gv.socketio.emit('reload')

def next_round_set():
    gv.parameters.clear()
    gv.post_num = 0
    
def global_model_update():
    if gv.model_name == 'ResNet':
        gv.model.load_parameter(gv.avg_weights)
        gv.model.model.float()
        train_loss, train_metric = gv.model.get_accuracy(gv.model.model, 'train')
        test_loss, test_metric = gv.model.get_accuracy(gv.model.model, 'test')
        val_loss, val_metric = gv.model.get_accuracy(gv.model.model, 'val') 
        print("global model val loss: %.6f, accuracy: %.2f %%" %(val_loss, 100*val_metric))
        accuracy = 100*val_metric

        gv.global_model_accuracy.append(100*val_metric)

    elif gv.model_name == 'YOLO':
        # gv.model.model.train()
        # gv.avg_weights = {k: v.clone() for k, v in gv.avg_weights.items()
        model = YOLO('yolov10s.yaml')
        model.model.load_state_dict(gv.avg_weights)
        model.model.float()
        yaml_path = "D:\\fed\\coco_yolo_clients\\client_0_data.yaml"
        
        with torch.no_grad():
            model.model.eval()
            results = model.val(data=yaml_path, split="test")  # yaml 파일을 통해 데이터 로드

        # 정확도 출력
        accuracy = results.box.map * 100  # mAP@0.5 (Mean Average Precision at 0.5 IOU)
        print(f"Model Accuracy (mAP@0.5): {accuracy:.2f}%")
        gv.global_model_accuracy.append(accuracy)
        
    print("wandb logging...")
    wandb.log({"test_loss" : test_loss, "test_acc" : test_metric, "val_loss" : val_loss, "val_acc" : val_metric, 'train_loss' : train_loss, 'train_acc' : train_metric})
    # wandb.log({"val_loss" : val_loss, "val_acc" : val_metric})

    
    gv.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if accuracy > gv.best_acc:
        gv.best_acc = accuracy
        gv.best_model_wts = gv.avg_weights
        gv.best_round = gv.round_num
        print("best model updated")