import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from LSTM import LSTM
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary

class AirQualityDataset(Dataset):
    def __init__(self, data, seq_length=24):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]
        target = self.data[idx + self.seq_length, 0]  # pollution 값만 예측
        return torch.FloatTensor(sequence), torch.FloatTensor([target])

def preprocess_data(file_path, seq_length=24):
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    # 날짜 컬럼 제외하고 필요한 특성 선택
    features = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
    df = df[features]
    
    # 결측치 처리
    df = df.ffill()
    
    # 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # 데이터셋 생성
    dataset = AirQualityDataset(scaled_data, seq_length)
    
    return dataset, scaler

def plot_metrics(train_metrics, val_metrics, save_path='training_metrics.png'):
    """
    학습 과정에서의 평가 지표들을 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['MSE', 'MAE', 'RMSE', 'Accuracy']
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        axes[row, col].plot(range(1, len(train_metrics[metric]) + 1), 
                          train_metrics[metric], label=f'Train {metric}', marker='o')
        axes[row, col].plot(range(1, len(val_metrics[metric]) + 1), 
                          val_metrics[metric], label=f'Val {metric}', marker='o')
        axes[row, col].set_xticks(range(1, len(train_metrics[metric]) + 1))
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel(metric)
        axes[row, col].set_title(f'{metric} over Training')
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_comparison(true_values, predictions, save_path='prediction_comparison.png'):
    """
    실제값과 예측값 비교 시각화
    """
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='True Values', marker='o')
    plt.plot(predictions, label='Predictions', marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Pollution Level')
    plt.title('Prediction vs True Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(predictions, targets, threshold=0.3):
    """
    회귀 평가 지표 계산: MSE, MAE, RMSE, Accuracy-like
    threshold: 실제값의 ±10% 범위 내에 예측값이 들어오면 correct로 간주
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    
    # accuracy_like 계산: 예측값이 실제값의 ±threshold 범위 내에 있는 비율
    within_range = torch.abs(predictions - targets) <= (threshold * torch.abs(targets))
    accuracy_like = torch.mean(within_range.float()).item() * 100  # 백분율로 변환
    
    return mse, mae, rmse, accuracy_like

def train(model, data_loader, criterion, optimizer, device):
    model.train()
    total_mse = 0
    total_mae = 0
    total_rmse = 0
    total_acc = 0
    n_batches = len(data_loader)
    
    for sequences, targets in data_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        sequences = sequences.transpose(0, 1)
        
        optimizer.zero_grad()
        output, _ = model(sequences)
        output = output[-1, :, 0]
        
        loss = criterion(output, targets.squeeze())
        loss.backward()
        optimizer.step()
        
        mse, mae, rmse, acc = calculate_metrics(output, targets.squeeze())
        total_mse += mse
        total_mae += mae
        total_rmse += rmse
        total_acc += acc
    
    return (total_mse / n_batches, 
            total_mae / n_batches, 
            total_rmse / n_batches,
            total_acc / n_batches)

def evaluate(model, data_loader, criterion, device, return_predictions=False):
    model.eval()
    total_mse = 0
    total_mae = 0
    total_rmse = 0
    total_acc = 0
    n_batches = len(data_loader)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            sequences = sequences.transpose(0, 1)
            
            output, _ = model(sequences)
            output = output[-1, :, 0]
            
            if return_predictions:
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(targets.squeeze().cpu().numpy())
            
            mse, mae, rmse, acc = calculate_metrics(output, targets.squeeze())
            total_mse += mse
            total_mae += mae
            total_rmse += rmse
            total_acc += acc
    
    metrics = (total_mse / n_batches, 
              total_mae / n_batches, 
              total_rmse / n_batches,
              total_acc / n_batches)
    
    if return_predictions:
        return metrics, (all_predictions, all_targets)
    return metrics

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    seq_length = 24
    batch_size = 32
    input_size = 7
    hidden_size = 64
    n_layers = 2
    num_epochs = 10
    learning_rate = 0.001
    
    # 데이터 준비
    # file_path = 'data/air_pollution/LSTM-Multivariate_pollution.csv'
    file_path = 'lstm_client_data/client_1_data.csv'
    dataset, scaler = preprocess_data(file_path, seq_length)
    
    # 학습/검증 데이터 분할
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    # 모델 초기화
    model = LSTM(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    summary(model, input_size=(seq_length, input_size))
    
    # # 학습 지표 저장을 위한 딕셔너리
    # train_metrics = {'MSE': [], 'MAE': [], 'RMSE': [], 'Accuracy': []}
    # val_metrics = {'MSE': [], 'MAE': [], 'RMSE': [], 'Accuracy': []}
    
    # # 학습
    # best_val_mse = float('inf')
    # for epoch in range(num_epochs):
    #     train_mse, train_mae, train_rmse, train_acc = train(model, train_loader, criterion, optimizer, device)
    #     val_mse, val_mae, val_rmse, val_acc = evaluate(model, test_loader, criterion, device)
        
    #     # 지표 저장
    #     train_metrics['MSE'].append(train_mse)
    #     train_metrics['MAE'].append(train_mae)
    #     train_metrics['RMSE'].append(train_rmse)
    #     train_metrics['Accuracy'].append(train_acc)
    #     val_metrics['MSE'].append(val_mse)
    #     val_metrics['MAE'].append(val_mae)
    #     val_metrics['RMSE'].append(val_rmse)
    #     val_metrics['Accuracy'].append(val_acc)
        
    #     if val_mse < best_val_mse:
    #         best_val_mse = val_mse
    #         torch.save(model.state_dict(), 'best_model.pth')
        
    #     print(f'Epoch [{epoch+1}/{num_epochs}]')
    #     print(f'Train - MSE: {train_mse:.6f}, MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, Acc: {train_acc:.2f}%')
    #     print(f'Val   - MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}, Acc: {val_acc:.2f}%')
    #     print('-' * 80)
    
    # # 학습 과정 시각화
    # plot_metrics(train_metrics, val_metrics)
    
    # # 최종 예측 성능 평가
    # model.load_state_dict(torch.load('best_model.pth'))
    # final_metrics, (predictions, targets) = evaluate(model, test_loader, criterion, device, return_predictions=True)
    
    # # 예측 결과 시각화 (처음 100개 샘플만)
    # plot_prediction_comparison(targets[:100], predictions[:100])
    
    # print("\n최종 평가 결과:")
    # print(f"MSE: {final_metrics[0]:.6f}")
    # print(f"MAE: {final_metrics[1]:.6f}")
    # print(f"RMSE: {final_metrics[2]:.6f}")
    # print(f"Accuracy: {final_metrics[3]:.2f}%")
    
    # # 예측 함수
    # def predict(model, data, scaler):
    #     model.eval()
    #     with torch.no_grad():
    #         data = torch.FloatTensor(data).unsqueeze(0).to(device)
    #         data = data.transpose(0, 1)
            
    #         output, _ = model(data)
    #         output = output[-1, :, 0].cpu().numpy()
            
    #         # 역정규화 (pollution 값만)
    #         prediction = scaler.inverse_transform(
    #             np.concatenate([output.reshape(-1, 1), 
    #                           np.zeros((output.shape[0], scaler.scale_.shape[0]-1))], 
    #                           axis=1))[:, 0]
            
    #         return prediction

    # # 예측 테스트
    # test_sequence = dataset.data[-seq_length:]
    # prediction = predict(model, test_sequence, scaler)
    # print(f"\n다음 시간의 pollution 예측값: {prediction[0]:.2f}")