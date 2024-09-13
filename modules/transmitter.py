import base64
import csv
import datetime
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
from modules.client_manager import send_reload_signal, update_status, send_training_signal

transmitter_lock = threading.Lock()

@transmitter_bp.route('/transmitter', methods=['POST', 'GET'])
def handle_parameter():

    if request.method == 'POST':  # 클라이언트로부터 파라미터 수신
        data = request.json
        client_name = data['client_name']
        print(f"params received from {client_name}")
        comp_data = base64.b64decode(data['params']) # base64 디코딩
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
            gv.train_mode = 'default'
            gv.auto_end_time = datetime.datetime.now()
            print("auto run complete")

            dtime = duration_of_time(gv.auto_start_time, gv.auto_end_time)
            # 실험 결과를 CSV 파일로 저장
            metadata = {
                "Client Count": len(gv.client_list),
                "Auto Run Rounds": gv.auto_run_rounds,
                "Clients Epochs(round)": 2,
                "model": gv.model.model_name,
                "Dataset": "CIFAR-10",
                "Data Size": gv.model.get_data_size(),
                "Best Round": gv.best_round,
                "Best Accuracy": gv.best_acc,
                "Duration of time": dtime
            }
            save_experiment_results_csv(len(gv.client_list), gv.global_model_accuracy, metadata)
            return "auto run complete"
        elif all_clients_same_signal(signal):
            if signal == 'ready':
                gv.round_num += 1
                # gv.socketio.emit('training')
                send_training_signal()
                print()
                print("="*10)
                print(f"round {gv.round_num} start")
                print("training signal sent")
                return "training signal sent"
            elif signal == 'Finish':
                msg = round_manager()
                # print(msg)
                # test_loss, test_metric = gv.model.get_accuracy(gv.model.model, 'test')
                # val_loss, val_metric = gv.model.get_accuracy(gv.model.model, 'val')

                # print("wandb logging...")
                # wandb.log({"test_loss" : test_loss, "test_acc" : test_metric, "val_loss" : val_loss, "val_acc" : val_metric})

                return msg

    return "signal received"

def duration_of_time(start_time, end_time):
    duration = end_time - start_time
    total_seconds = duration.total_seconds()
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    # "000시간 00분 00초" 형식으로 포맷팅 (시간은 필요에 따라 자릿수 확장)
    formatted_duration = f"{hours} h {minutes:02} min {seconds:02} sec"
    
    return formatted_duration

def all_clients_same_signal(signal):
    for status in gv.client_status.values():
        if status != signal:
            return False
    return True

def save_experiment_results_csv(client_count, round_accuracies, experiment_metadata):
    """
    실험 결과를 CSV 파일로 저장하는 함수.

    parameter:
    - client_count: 참여 클라이언트 수
    - round_accuracies: 라운드 별 글로벌 모델 정확도 리스트
    - experiment_metadata: 실험 환경에 대한 메타데이터 (예: 학습률, 에포크 수 등)

    """
    # 파일을 저장할 디렉토리 설정
    output_dir = "experiment_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 디렉토리의 파일 개수를 확인하여 새로운 파일 이름 생성
    file_count = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
    new_file_name = f"non_iid_experiment_results_{file_count + 1}.csv"

    # CSV 파일로 저장
    output_file = os.path.join(output_dir, new_file_name)
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # 메타데이터를 첫 번째 줄에 기록
        csv_writer.writerow(["Experiment Metadata"])
        for key, value in experiment_metadata.items():
            csv_writer.writerow([key, value])
        csv_writer.writerow([])  # 빈 줄 추가
        
        # 헤더 작성
        csv_writer.writerow(["Round", "Accuracy (%)", "Client Count"])
        
        # 각 라운드 별 정확도와 클라이언트 수를 기록
        for round_num, accuracy in enumerate(round_accuracies, start=1):
            csv_writer.writerow([round_num, accuracy, client_count])

    print(f"Experiment results saved to {output_file}")