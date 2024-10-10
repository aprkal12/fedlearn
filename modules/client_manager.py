import csv
import datetime
import os
from flask import current_app, request, g
from flask_socketio import join_room
from . import client_bp
import zstd
import pickle
from models.Resnet_infer import Inference
import global_vars as gv
import datetime


@client_bp.route('/client', methods=['POST', 'GET'])
def register_client():
    if request.method == 'POST':
        data = request.json  # 클라이언트로부터 전송된 JSON 데이터 받기

        # client_name = data['id']
        client_name = data['sid']
        # client_id = f"client_{len(gv.client_list)+1}" # 이건 현재 클라이언트 수
        client_id =  f"client_{gv.client_len+1}" # 이건 지금까지의 모든 클라이언트 수(중간에 나갔어도)
        
        gv.client_len += 1
        gv.client_list[client_id] = client_name
        # gv.client_status[client_name] = 'waiting'
        # gv.client_sockets[client_id] = data['sid']
        
        gv.client_status[client_id] = 'waiting'
        send_reload_signal()
        return client_id
    
    elif request.method == 'GET':
        params = gv.model.parameter_extract()
        binary_data = pickle.dumps(params)
        comp_data = zstd.compress(binary_data)

        return comp_data
    

@client_bp.route('/client/training', methods=['POST'])
def training():
    count = 0
    start = len(gv.client_list)
    for client in gv.client_list:
        if gv.client_status[client] == 'ready' or gv.client_status[client] == 'join':
            count += 1
        else:
            break

    if count == start:
        # gv.socketio.emit('training')
        send_training_signal()
        gv.round_num += 1
        print()
        print("="*10)
        print("round %d start" % gv.round_num)
        print("training signal sent")
        return "training signal sent"
    else:
        print("not all clients are ready")
        return "not all clients are ready"
    
@client_bp.route('/client/autorun', methods=['POST', 'DELETE'])
def auto_run():
    if request.method == 'POST':
        if gv.train_mode == 'auto':
            print("auto run already started")
            return "auto run already started"
        else:
            gv.auto_run_rounds = int(request.args.get('rounds', 1))  # 사용자가 지정한 라운드 수 가져오기
            gv.auto_start_time = datetime.datetime.now()  # 자동 학습 시작 시간 기록
            gv.train_mode = 'auto'  # 학습 모드를 'auto'로 설정
            print("auto run start")
            print("goal rounds: ", gv.auto_run_rounds)
            if gv.train_mode == 'auto':
                # gv.socketio.emit('training')
                send_training_signal()
                gv.round_num += 1
                print()
                print("="*10)
                print("round %d start" % gv.round_num)
                print("training signal sent")
            return "auto run start"
    elif request.method == 'DELETE':
        autorun_complete()
        print("auto run stopped")
        return "auto run stopped"

@client_bp.route('/client/disconnect', methods=['POST'])
def disconnect_client():
    client_id = request.args.get('client_id')  # 요청에서 클라이언트 ID 가져오기
    print(client_id)
    print(gv.client_list)
    if client_id in gv.client_list:
        # 소켓 ID를 사용하여 클라이언트 연결 해제
        sid = gv.client_list[client_id]
        # gv.socketio.disconnect(sid)  # 소켓 연결 해제
        # print(f"Emitting test_disconnect to room: {sid}")
        # gv.socketio.emit('test_disconnect', room=sid)
        send_disconnect_signal(sid)

        # 클라이언트 목록 및 소켓 정보에서 제거
        del gv.client_list[client_id]
        del gv.client_status[client_id]

        send_reload_signal()  # UI 갱신 신호 전송

        return f"Client {client_id} disconnected successfully.", 200
    else:
        return "Client not found.", 404

def autorun_complete():
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
    send_reload_signal()

def duration_of_time(start_time, end_time):
    duration = end_time - start_time
    total_seconds = duration.total_seconds()
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    # "000시간 00분 00초" 형식으로 포맷팅 (시간은 필요에 따라 자릿수 확장)
    formatted_duration = f"{hours} h {minutes:02} min {seconds:02} sec"
    
    return formatted_duration

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

### 소켓 통신 담당 ###
def send_disconnect_signal(sid):
    gv.socketio.emit('disconnect', to=sid)

def send_aggregated_signal():
    gv.socketio.emit('aggregated_params')

def send_reload_signal():
    gv.socketio.emit('reload')

def update_status(name, signal):
    gv.client_status[name] = signal
    gv.socketio.emit('update_status', {'name': name, 'signal': signal})

def send_training_signal(sid=None):
    if sid:
        gv.socketio.emit('training', to=sid)
    else:
        gv.socketio.emit('training')

def update_data(data):
    gv.socketio.emit('update_data', data)

