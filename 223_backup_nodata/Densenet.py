import torch
from torch import nn
from torchinfo import summary

class Bottleneck(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4 * k, 1, bias = False),
            nn.BatchNorm2d(4 * k),
            nn.ReLU(),
            nn.Conv2d(4 * k, k, 3, padding = 1, bias = False),
        )
        
    def forward(self, x):
        return torch.cat([self.residual(x), x], dim = 1)
    
class Transition(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels), # Dense block end: Conv
            nn.ReLU(),
            nn.Conv2d(in_channels, int(in_channels / 2), 1, bias = False), # Reduce channels
            nn.AvgPool2d(2),    # Reduce feature map size
        )
    
    def forward(self, x):
        return self.transition(x)

class DenseNet(nn.Module):
    def __init__(self, block_list, growth_rate, n_classes = 1000):
        super().__init__()

        assert len(block_list) == 4

        self.k = growth_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 2 * self.k, 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(2 * self.k),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, stride = 2, padding = 1)

        self.dense_channels = 2 * self.k
        dense_blocks = []
        dense_blocks.append(self.make_dense_block(block_list[0]))
        dense_blocks.append(self.make_dense_block(block_list[1]))
        dense_blocks.append(self.make_dense_block(block_list[2]))
        dense_blocks.append(self.make_dense_block(block_list[3], last_stage = True))
        self.dense_blocks = nn.Sequential(*dense_blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.dense_channels, n_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.dense_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def make_dense_block(self, num_blocks, last_stage = False):
        layers = []
        for _ in range(num_blocks):
            layer = Bottleneck(self.dense_channels, self.k)
            layers.append(layer)
            self.dense_channels += self.k

        if last_stage:
            layers.append(nn.BatchNorm2d(self.dense_channels))
            layers.append(nn.ReLU())
        else:
            layers.append(Transition(self.dense_channels))
            assert self.dense_channels % 2 == 0
            self.dense_channels //= 2
            
        return nn.Sequential(*layers)

def DenseNet121():
    return DenseNet(block_list = [6, 12, 24, 16], growth_rate = 32)

def DenseNet169():
    return DenseNet(block_list = [6, 12, 32, 32], growth_rate = 32)

def DenseNet201():
    return DenseNet(block_list = [6, 12, 48, 32], growth_rate = 32)

def DenseNet264():
    return DenseNet(block_list = [6, 12, 64, 48], growth_rate = 32)
    
    
if __name__ == '__main__':
    model = DenseNet121()
    summary(model, input_size = (2, 3, 224, 224))
    # # 모델 생성 및 손실 함수, 최적화기 설정
    # model = DenseNet121()  # 또는 DenseNet169(), DenseNet201(), DenseNet264()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # 학습 실행
    # train_model(model, train_loader, criterion, optimizer, num_epochs=10)