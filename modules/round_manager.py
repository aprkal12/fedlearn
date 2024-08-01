from flask import current_app, request, g
from flask_socketio import emit, SocketIO
from . import aggregate_bp
import threading
import torch
from models.Resnet_infer import Inference
import global_vars as gv

parameter_lock = threading.Lock()
expected_clients = 2

@aggregate_bp.route('/aggregate', methods=['POST'])
def round_manager():
    gv.round_num += 1
    print("라운드 %d 시작" % gv.round_num)
    return aggregate_parameters()


def aggregate_parameters():
    global expected_clients

    print("post_num : ", gv.post_num)
    print("expected_clients : ", expected_clients)
    with parameter_lock:
        if gv.post_num == expected_clients:
            gv.avg_weights = {}
            tensorlist_float = [{key: value.float() for key, value in client_weights.items()} for client_weights in gv.parameters]
            for key in tensorlist_float[0].keys():
                gv.avg_weights[key] = torch.stack([client_weights[key] for client_weights in tensorlist_float], dim=0).mean(dim=0)
            
            gv.model.load_parameter(gv.avg_weights)
            print("모든 파라미터 수신 완료 및 평균 파라미터 계산 완료")
            val_loss, val_metric = gv.model.get_accuracy(gv.model.model)
            print("val loss: %.6f, accuracy: %.2f %%" %(val_loss, 100*val_metric))
            gv.parameters.clear()
            gv.post_num = 0

            gv.global_model_status[gv.round_num] = 100*val_metric
            gv.socketio.emit('aggregated_params')
            notify_clients()
            return "집계 완료"
        else:
            print("집계 조건 충족 안됨")
    return "아직 모든 파라미터가 수신되지 않았습니다."

def notify_clients():
    gv.socketio.emit('reload')
