import torch
import torch.nn as nn
# 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 입력 채널: 3 (RGB 이미지), 출력 채널: 16, 커널 크기: 3x3, 패딩: 1
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  # 입력 채널: 16, 출력 채널: 32, 커널 크기: 3x3, 패딩: 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Max Pooling, 크기 2x2, 스트라이드 2
            nn.Conv2d(32, 64, 3, padding=1),  # 입력 채널: 32, 출력 채널: 64, 커널 크기: 3x3, 패딩: 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Max Pooling, 크기 2x2, 스트라이드 2
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 8 * 8, 100),  # Fully Connected Layer, 입력 크기: 64*8*8, 출력 크기: 100
            nn.ReLU(),
            nn.Linear(100, 10)  # Fully Connected Layer, 입력 크기: 100, 출력 크기: 10 (클래스 개수)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size(0), -1)  # Flatten 작업
        out = self.fc_layer(out)
        return out