from flask_socketio import SocketIO

post_num = 0 # 클라이언트에서 받은 파라미터 수

client_list = {} # 클라이언트 리스트

model = None # 모델 객체

avg_weights = None # 평균 파라미터

parameters = {} # 클라이언트에서 받은 파라미터들

socketio = SocketIO(ping_timeout=120, cors_allowed_origins="*") # 소켓 객체

round_num = 0 # 현재 라운드

global_model_accuracy = [] # 글로벌 모델 정확도

client_status = {} # 클라이언트 상태

train_mode = 'default' # 학습 모드

auto_run_rounds = 1 # 자동 학습 라운드 수

auto_start_time = None # 자동 학습 시작 시간

auto_end_time = None # 자동 학습 종료 시간

best_acc = 0.0 # 최고 정확도

best_round = 0 # 최고 정확도 라운드

best_model_wts = None # 최고 정확도 모델 가중치

last_updated = None # 마지막 업데이트 시간

client_len = 0 # 클라이언트 수 (중간에 클라이언트가 나가도 유지)

# client_sockets = {} # 클라이언트 소켓