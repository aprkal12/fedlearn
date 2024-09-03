import datetime
from flask import current_app, request, g
from . import client_bp
import zstd
import pickle
from models.Resnet_infer import Inference
import global_vars as gv


@client_bp.route('/client', methods=['POST'])
def register_client():
    if request.method == 'POST':
        data = request.json  # 클라이언트로부터 전송된 JSON 데이터 받기

        client_name = data['id']
        client_id = f"client_{len(gv.client_list)+1}"

        gv.client_list[client_id] = client_name
        print("client registered")
        gv.socketio.emit('reload')
        
        params = gv.model.parameter_extract()
        binary_data = pickle.dumps(params)
        comp_data = zstd.compress(binary_data)

        return comp_data
    

@client_bp.route('/client/training', methods=['POST'])
def training():
    count = 0
    start = len(gv.client_list)
    for client in gv.client_list:
        if gv.client_status[client] == 'ready':
            count += 1
        else:
            break

    if count == start:
        gv.socketio.emit('training')
        gv.round_num += 1
        print()
        print("="*10)
        print("round %d start" % gv.round_num)
        print("training signal sent")
        return "training signal sent"
    else:
        print("not all clients are ready")
        return "not all clients are ready"
    
@client_bp.route('/client/auto_run', methods=['POST'])
def auto_run():
    gv.auto_run_rounds = int(request.args.get('rounds', 1))  # 사용자가 지정한 라운드 수 가져오기
    gv.auto_start_time = datetime.datetime.now()  # 자동 학습 시작 시간 기록
    gv.train_mode = 'auto'  # 학습 모드를 'auto'로 설정
    print("auto run start")
    print("goal rounds: ", gv.auto_run_rounds)
    if gv.round_num == 0:
        gv.socketio.emit('training')
        gv.round_num += 1
        print()
        print("="*10)
        print("round %d start" % gv.round_num)
        print("training signal sent")
    return "auto run start"

@client_bp.route('/client/name', methods=['POST'])
def get_name():
    data = request.data.decode('utf-8')
    name = [k for k, v in gv.client_list.items() if v == data][0]
    return name

