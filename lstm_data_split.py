import pandas as pd
import numpy as np
import os

def split_data_for_clients(file_path, num_clients, output_dir='lstm_client_data'):
    """
    데이터를 클라이언트 수에 맞게 균등하게 분할하고 각각 저장
    
    Parameters:
    - file_path: 원본 데이터 파일 경로
    - num_clients: 클라이언트 수
    - output_dir: 분할된 데이터를 저장할 디렉토리
    """
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    # 전체 데이터 수
    total_rows = len(df)
    
    # 각 클라이언트당 데이터 수 계산
    rows_per_client = total_rows // num_clients
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 분할 및 저장
    for i in range(num_clients):
        start_idx = i * rows_per_client
        
        # 마지막 클라이언트는 남은 모든 데이터를 가져감
        if i == num_clients - 1:
            end_idx = total_rows
        else:
            end_idx = (i + 1) * rows_per_client
        
        # 클라이언트별 데이터 추출
        client_df = df.iloc[start_idx:end_idx].copy()
        
        # 데이터 저장
        output_path = os.path.join(output_dir, f'client_{i+1}_data.csv')
        client_df.to_csv(output_path, index=False)
        
        # 클라이언트별 데이터 정보 출력
        print(f'Client {i+1}:')
        print(f'  - Data rows: {len(client_df)}')
        print(f'  - Data shape: {client_df.shape}')
        print(f'  - Saved to: {output_path}')
        print('-' * 50)
    
    print(f'\nTotal data rows: {total_rows}')
    print(f'Rows per client: {rows_per_client}')
    print(f'Data successfully split into {num_clients} parts')

def verify_data_split(output_dir='client_data'):
    """
    분할된 데이터의 통계 정보를 확인
    """
    total_rows = 0
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])
    
    print("\nData Split Verification:")
    print("-" * 50)
    
    for file in files:
        file_path = os.path.join(output_dir, file)
        df = pd.read_csv(file_path)
        total_rows += len(df)
        
        print(f'File: {file}')
        print(f'Number of rows: {len(df)}')
        print(f'Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB')
        print('-' * 50)
    
    print(f'Total rows across all splits: {total_rows}')

if __name__ == "__main__":
    # 설정
    data_file = 'data/air_pollution/LSTM-Multivariate_pollution.csv'
    num_clients = 5  # 클라이언트 수를 여기서 설정
    output_dir = 'lstm_client_data'
    
    # 데이터 분할
    print(f"Splitting data for {num_clients} clients...\n")
    split_data_for_clients(data_file, num_clients, output_dir)
    
    # 분할 결과 검증
    verify_data_split(output_dir)