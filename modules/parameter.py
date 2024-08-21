import base64
from flask import current_app, request, g
from . import parameter_bp
import zstd
import pickle
import threading
import global_vars as gv

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
        
        # print(gv.parameters.keys())
        

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
    return "signal received"
