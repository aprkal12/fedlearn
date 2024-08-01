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
        comp_data = request.data
        decomp_data = zstd.decompress(comp_data)
        client_params = pickle.loads(decomp_data)
        print("파라미터 수신 확인")
        with parameter_lock:
            gv.parameters.append(client_params)
            gv.post_num += 1
            print("post_num : ", gv.post_num)

        return "서버 : 파라미터 전송 완료"

    elif request.method == 'GET':
        if gv.avg_weights is None: # avg_wights도 전역변수 선언해야함
            return "이번 라운드의 평균 파라미터가 아직 집계되지 않았습니다.", 400
        binary_data = pickle.dumps(gv.avg_weights)
        comp_data = zstd.compress(binary_data)
        return comp_data
