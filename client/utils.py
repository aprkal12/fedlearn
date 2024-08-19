import socket

hostname = None
hostip = None

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
