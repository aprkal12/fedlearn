import wandb
import socket
import requests
import re
from refac import aggregate_param

# in_addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# in_addr.connect(("www.google.com", 443))
# print("내부 아이피 : ", in_addr.getsockname()[0])

# req = requests.get("http://checkip.dyndns.org")
# external_ip = req.text.split(": ")[1]
# external_ip = re.sub('<.+?>', '', external_ip)
# print("외부 아이피 : ", external_ip)


aggregate_param()





# Wandb 프로젝트 이름과 사용자 이름을 설정합니다.
# wandb_project = "Federated_Learning"
# wandb_entity = "aprkal12"

# # Wandb API를 사용하여 특정 프로젝트의 모든 런(run) 정보를 가져옵니다.
# api = wandb.Api()
# runs = api.runs(f"{wandb_entity}/{wandb_project}")

# # 가져온 런(run) 정보를 출력합니다.
# # for run in runs:
# #     print(run.files())
# files = runs[0].files()
# for file in files:
#     print(file)
