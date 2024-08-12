import socket

def set_host():
    hostname = socket.gethostname()
    hostip = socket.gethostbyname(hostname)
    return hostname, hostip

