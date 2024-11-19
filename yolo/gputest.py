import torch

# 현재 GPU 장치 정보 출력
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(device)
compute_capability = torch.cuda.get_device_capability(device)

# bfloat16 지원 여부 확인
supports_bfloat16 = torch.cuda.get_device_properties(device).major >= 8  # A100처럼 Compute Capability 8.0 이상인 경우 지원 가능성 있음

print(f"Device Name: {device_name}")
print(f"Compute Capability: {compute_capability}")
print(f"Supports bfloat16: {supports_bfloat16}")

if supports_bfloat16:
    print("This GPU supports bfloat16 operations.")
else:
    print("This GPU does NOT support bfloat16 operations.")
