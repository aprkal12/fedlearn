import pandas as pd

# CSV 파일 경로 (다운로드한 데이터셋 파일의 경로를 설정하세요)
file_path = 'data/air_pollution/LSTM-Multivariate_pollution.csv'

# CSV 파일 로드
df = pd.read_csv(file_path)

# 데이터 확인
print("데이터 미리보기:")
print(df.columns)  # 상위 5개 데이터 확인

# 데이터 개수 확인
print("\n총 데이터 수:", len(df))

# # 기본 통계 정보
# print("\n기본 통계 정보:")
# print(df.describe())
