from flask import Flask, current_app, render_template, g
from flask_socketio import SocketIO
from modules import parameter_bp, aggregate_bp, client_bp, trigger_bp
from Resnet_infer import Inference
import global_vars as gv


app = Flask(__name__)
gv.socketio = SocketIO(app)

app.register_blueprint(parameter_bp)
app.register_blueprint(aggregate_bp)
app.register_blueprint(client_bp)
app.register_blueprint(trigger_bp)


@app.before_request
def before_request():
    print("before_request")
    
    # if 'client_list' not in g:
    #     g.client_list = {}
    # if 'model' not in g:
    #     g.model = model
    # if 'post_num' not in g:
    #     g.post_num = 0

@app.route('/')
def mainpage():
    return render_template('index.html', clients=gv.client_list)

if __name__ == '__main__':

    gv.model = Inference()
    gv.model.set_variable()
    gv.model.set_epoch(1)
    gv.model.run()
    print("모델 학습 완료")
    datas = gv.model.parameter_extract()  # 모델에서 파라미터 추출
    print("초기 파라미터 추출 완료")
    print("서버 실행 중")
    gv.socketio.run(app, host='0.0.0.0', port=11110)
