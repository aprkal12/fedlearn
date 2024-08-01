from flask_socketio import SocketIO

post_num = 0 # 클라이언트에서 받은 파라미터 수

client_list = {} # 클라이언트 리스트

model = None # 모델 객체

avg_weights = None # 평균 파라미터

parameters = [] # 클라이언트에서 받은 파라미터들

socketio = SocketIO() # 소켓 객체