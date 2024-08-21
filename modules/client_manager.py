from flask import current_app, request, g
from . import client_bp
import zstd
import pickle
from models.Resnet_infer import Inference
import global_vars as gv


@client_bp.route('/client', methods=['POST'])
def register_client():

    data = request.json  # 클라이언트로부터 전송된 JSON 데이터 받기
    client_name = data['hostname']
    client_ip = data['ip']
    gv.client_list[client_name] = client_ip
    print("클라이언트 접속 확인")
    print(gv.client_list)
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


def unregister_client(client_name):
    if client_name in gv.client_list:
        del gv.client_list[client_name]
        print("클라이언트 삭제 확인")
        print(gv.client_list)
        return "클라이언트 삭제 완료"


