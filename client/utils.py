import socket
import uuid

hostname = None
hostip = None
uid = None
name = None
sid = None

def set_host():
    global hostname, hostip
    hostname = socket.gethostname()
    hostip = socket.gethostbyname(hostname)

def get_hostname():
    global hostname
    return hostname

def get_hostip():
    global hostip
    return hostip

def get_host():
    global hostname, hostip
    return hostname, hostip

# 파일 크기 mb 변환
def bytes_to_mb(size_in_bytes):
    return size_in_bytes / (1024 * 1024)

def set_id():
    global uid
    uid = str(uuid.uuid4())

def get_id():
    global uid
    return uid

def set_name(received_name):
    global name
    name = received_name

def get_name():
    global name
    return name

def set_sid(received_sid):
    global sid
    sid = received_sid

def get_sid():
    global sid
    return sid