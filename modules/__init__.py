from flask import Blueprint

# 블루프린트 객체 생성
parameter_bp = Blueprint('parameter', __name__)
aggregate_bp = Blueprint('aggregate', __name__)
client_bp = Blueprint('client', __name__)

# 각 블루프린트 모듈에서 라우트를 임포트
from .parameter import *
from .round_manager import *
from .client_manager import *
