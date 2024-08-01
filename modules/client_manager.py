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

    params = gv.model.parameter_extract()
    binary_data = pickle.dumps(params)
    comp_data = zstd.compress(binary_data)

    return comp_data

def unregister_client(client_name):
    if client_name in gv.client_list:
        del gv.client_list[client_name]
        print("클라이언트 삭제 확인")
        print(gv.client_list)
        return "클라이언트 삭제 완료"
