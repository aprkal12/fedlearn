import base64
import csv
import json
import os
from flask import current_app, request, g
from . import parameter_bp
import zstd
import pickle
import threading
import global_vars as gv

from modules.round_manager import round_manager

parameter_lock = threading.Lock()

@parameter_bp.route('/parameter', methods=['POST', 'GET'])
def handle_parameters():

    if request.method == 'POST':  # 클라이언트로부터 파라미터 수신
        data = request.json
        client_name = data['client_name']
        print(f"params received from {client_name}")
        comp_data = base64.b64decode(data['params']) # base64 디코딩
        # comp_data = bytes.fromhex(data['params']) # hex 디코딩
        # comp_data = request.data
        decomp_data = zstd.decompress(comp_data)
        client_params = pickle.loads(decomp_data)
        with parameter_lock:
            # gv.parameters.append(client_params)
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

@parameter_bp.route('/parameter/signal', methods=['POST'])
def signal():
    data = request.json
    name = data['name']
    signal = data['signal']
    
    print(f"signal received -> {name} : {signal}")

    gv.client_status[name] = signal
    gv.socketio.emit('update_status', {'name': name, 'signal': signal})
    gv.socketio.emit('reload')

    if gv.round_num == gv.auto_run_rounds:
        gv.train_mode = 'default'
        print("auto run complete")

        # 실험 결과를 CSV 파일로 저장
        metadata = {
            "Client Count": len(gv.client_list),
            "Auto Run Rounds": gv.auto_run_rounds,
            "Clients Epochs(round)": gv.model.epochs,
            "model": gv.model.model_name,
            "batch_size": gv.model.batch_size,
            "Learning Rate": gv.model.learning_rate,
            "Dataset": "CIFAR-10",
            "Data Size": gv.model.get_data_size()
        }
        save_experiment_results_csv(len(gv.client_list), gv.global_model_accuracy, metadata)
        return "auto run complete"
    
    if gv.train_mode == 'auto' and all_clients_same_signal(signal):
        if signal == 'ready':
            gv.round_num += 1
            gv.socketio.emit('training')
            print()
            print("="*10)
            print(f"round {gv.round_num} start")
            print("training signal sent")
            return "training signal sent"
        elif signal == 'Finish':
            msg = round_manager()
            print(msg)
            return msg

                
    return "signal received"

def all_clients_same_signal(signal):
    for status in gv.client_status.values():
        if status != signal:
            return False
    return True

def save_experiment_results_csv(client_count, round_accuracies, experiment_metadata):
    """
    실험 결과를 CSV 파일로 저장하는 함수.

    Parameters:
    - client_count: 참여 클라이언트 수
    - round_accuracies: 라운드 별 글로벌 모델 정확도 리스트
    - experiment_metadata: 실험 환경에 대한 메타데이터 (예: 학습률, 에포크 수 등)

    Returns:
    None
    """
    # 파일을 저장할 디렉토리 설정
    output_dir = "experiment_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 디렉토리의 파일 개수를 확인하여 새로운 파일 이름 생성
    file_count = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
    new_file_name = f"experiment_results_{file_count + 1}.csv"

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