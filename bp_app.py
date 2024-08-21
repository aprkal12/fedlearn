from flask import Flask, current_app, render_template, g
from flask_socketio import SocketIO
from modules import parameter_bp, aggregate_bp, client_bp
from models.Resnet_infer import Inference
import global_vars as gv
import datetime

app = Flask(__name__)
gv.socketio = SocketIO(app)

app.register_blueprint(parameter_bp)
app.register_blueprint(aggregate_bp)
app.register_blueprint(client_bp)

@gv.socketio.on('request_update')
def handle_request_update():
    # print("web_reloading")
    round_num = gv.round_num
    clients = gv.client_list
    client_status = gv.client_status # 클라이언트 상태
    clients_num = len(clients)

    if round_num == 0:
        clients_num = 0
        for client in clients:
            if client not in client_status.keys(): # 상태에 대한 정보가 없다면
                client_status[client] = 'waiting'

    if not gv.global_model_accuracy:
        gv.global_model_accuracy.append(0.0)
        global_model = gv.global_model_accuracy[0]
    elif len(gv.global_model_accuracy) < round_num+1:
        global_model = gv.global_model_accuracy[round_num-1] # 현재 라운드 학습중 (이전 라운드 정확도 표시)
    else:
        global_model = gv.global_model_accuracy[round_num]
    
    data = {
        'global_model_accuracy': global_model,  
        'current_round': round_num,             
        'clients': clients,
        'client_num': clients_num,
        'client_status': client_status,
        'rounds': list(range(0, len(gv.global_model_accuracy))),  # 라운드 숫자 리스트 생성
        'accuracy_history': gv.global_model_accuracy,  # 정확도 히스토리
        'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    gv.socketio.emit('update_data', data)

@app.route('/')
def mainpage():
    return render_template('index.html', clients=gv.client_list)

if __name__ == '__main__':
    gv.model = Inference()
    gv.model.set_variable(0.5)
    gv.model.set_epoch(1)
    gv.model.run()
    print("="*10)
    print("Start server")
    print("Model training complete")
    datas = gv.model.parameter_extract()  # 모델에서 파라미터 추출
    print("First params extracted")
    print("Server ready")
    gv.socketio.run(app, host='0.0.0.0', port=11110)
