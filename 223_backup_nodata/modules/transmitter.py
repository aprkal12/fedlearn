import base64
import csv

import json
import os
from flask import current_app, request, g
import wandb
from . import transmitter_bp
import zstd
import pickle
import threading
import global_vars as gv

from modules.global_model_manager import round_manager
from modules.client_manager import send_reload_signal, update_status, send_training_signal, autorun_complete

transmitter_lock = threading.Lock()

@transmitter_bp.route('/transmitter', methods=['POST', 'GET'])
def handle_parameter():

    if request.method == 'POST':  # 클라이언트로부터 파라미터 수신
        client_name = request.args.get('name', 'default')
        comp_data = request.data
        # data = request.json
        # client_name = data['client_name']
        if client_name not in gv.client_list.keys():
            return "client not registered", 400
        print(f"params received from {client_name}")
        # comp_data = base64.b64decode(data) # base64 디코딩
        # comp_data = bytes.fromhex(data['params']) # hex 디코딩
        # comp_data = request.data
        
        decomp_data = zstd.decompress(comp_data)
        client_params = pickle.loads(decomp_data)
        
        with transmitter_lock:
            gv.parameters[client_name] = client_params
            gv.post_num += 1

        return "server received params"

    elif request.method == 'GET':
        if gv.avg_weights is None: # avg_wights도 전역변수 선언해야함
            return "이번 라운드의 평균 파라미터가 아직 집계되지 않았습니다.", 400
            # return "The average parameters for this round have not been aggregated yet.", 400
        binary_data = pickle.dumps(gv.avg_weights)
        comp_data = zstd.compress(binary_data)
        return comp_data

@transmitter_bp.route('/transmitter/signal', methods=['POST'])
def signal():
    data = request.json
    name = data['name']
    signal = data['signal']
    
    print(f"signal received -> {name} : {signal}")

    gv.client_status[name] = signal
    update_status(name, signal)
    send_reload_signal()
    # gv.socketio.emit('update_status', {'name': name, 'signal': signal})
    # gv.socketio.emit('reload')

    if gv.train_mode == 'auto':
        if len(gv.global_model_accuracy) == gv.auto_run_rounds: 
            autorun_complete()
            return "auto run complete"
        elif all_clients_same_signal(signal):
            if signal == 'ready' or signal == 'join': # 조인은 첫번째 라운드 일 경우(클라이언트가 초기 파라미터만 가지고 있을 때)
                gv.round_num += 1
                # gv.socketio.emit('training')
                send_training_signal()
                print()
                print("="*10)
                print(f"round {gv.round_num} start")
                print("training signal sent")
                return "training signal sent"
            elif signal == 'finish':
                msg = round_manager()
                # print(msg)
                # test_loss, test_metric = gv.model.get_accuracy(gv.model.model, 'test')
                # val_loss, val_metric = gv.model.get_accuracy(gv.model.model, 'val')

                # print("wandb logging...")
                # wandb.log({"test_loss" : test_loss, "test_acc" : test_metric, "val_loss" : val_loss, "val_acc" : val_metric})

                return msg
        else:
            if signal == 'join':
                sid = gv.client_list[name]
                send_training_signal(sid)
                print(f"training signal sent to {name}")
                return f"training signal sent to {name}"
 
    return "signal received"


def all_clients_same_signal(signal):
    for status in gv.client_status.values():
        if status != signal:
            return False
    return True

