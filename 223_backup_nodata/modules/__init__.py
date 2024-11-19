from flask import Blueprint

# 블루프린트 객체 생성
transmitter_bp = Blueprint('transmitter', __name__)
aggregate_bp = Blueprint('aggregate', __name__)
client_bp = Blueprint('client', __name__)

# 각 블루프린트 모듈에서 라우트를 임포트
from .transmitter import *
from .global_model_manager import *
from .client_manager import *
