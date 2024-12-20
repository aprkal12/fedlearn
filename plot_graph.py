import matplotlib.pyplot as plt

# IID 및 Non-IID 데이터를 위한 Dictionary
iid_data = {
    "2 clients": [66.8, 78.1, 80.8, 81.7, 84.7, 86.5, 86.5, 86.5, 87.6, 87, 85, 88.6, 87.1, 88],
    "3 clients": [64.6, 71.7, 77.2, 80.4, 83.2, 83.9, 86, 84.4, 86, 86.6, 84.7, 85.9, 86.3, 87.8],
    "5 clients": [62.4, 68.6, 72.8, 76.2, 77.7, 80.4, 81.5, 81.5, 82.9, 81.8, 84.1, 84.1, 83.2, 84.7]
}

non_iid_data_trimmed = {
    "2 clients": [68.6, 77.9, 82.7, 84.1, 84.7, 85.9, 86.4, 86.9, 85.9, 87.2, 86.8, 86.4, 88.2, 88.2],
    "3 clients": [63, 73.5, 79.5, 80.8, 84.4, 85.4, 85.6, 85.8, 87.3, 86.4, 87.5, 86.3, 88.1, 88.9],
    "5 clients": [73.3, 77.4, 79.1, 80.3, 81, 82.2, 82, 82.1, 82.8, 82.9, 83, 83.5, 84.1, 83.5]
}

# 색상 설정
colors = {
    "2 clients": "blue",
    "3 clients": "green",
    "5 clients": "gold",
    "Centralized model": "red"
}

# Centralized model 정확도
centralized_accuracy_value = 88.2  # 88.2%
centralized_accuracy = [centralized_accuracy_value] * 14

# 사용자 지정 x축: 1부터 10까지는 연속 간격, 이후 20, 30, 40, 50 설정
x_positions = list(range(1, 11)) + [12, 14, 16, 18]
x_labels = [str(x) for x in range(1, 11)] + ["20", "30", "40", "50"]

# 그래프 생성
fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 세로 크기 축소

# IID 환경 그래프
for label, data in iid_data.items():
    axs[0].plot(x_positions, data, label=label, color=colors[label])

# Centralized model 추가
axs[0].plot(x_positions, centralized_accuracy, label="Centralized model", linestyle="-", color=colors["Centralized model"], linewidth=3)

# Centralized model 정확도 표시선 추가 (88.2로 표시)
axs[0].axhline(y=centralized_accuracy_value, color=colors["Centralized model"], linestyle="-", linewidth=2)
axs[0].text(0, centralized_accuracy_value, f"{centralized_accuracy_value}", color=colors["Centralized model"], va="bottom", ha="right")

axs[0].set_title("IID Environment")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Accuracy")
axs[0].set_xticks(x_positions)
axs[0].set_xticklabels(x_labels)  # 사용자 정의 레이블 적용
axs[0].legend()

# Non-IID 환경 그래프
for label, data in non_iid_data_trimmed.items():
    axs[1].plot(x_positions, data, label=label, color=colors[label])

# Centralized model 추가
axs[1].plot(x_positions, centralized_accuracy, label="Centralized model", linestyle="-", color=colors["Centralized model"], linewidth=3)

# Centralized model 정확도 표시선 추가 (88.2로 표시)
axs[1].axhline(y=centralized_accuracy_value, color=colors["Centralized model"], linestyle="-", linewidth=2)
axs[1].text(0, centralized_accuracy_value, f"{centralized_accuracy_value}", color=colors["Centralized model"], va="bottom", ha="right")

axs[1].set_title("Non-IID Environment")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].set_xticks(x_positions)
axs[1].set_xticklabels(x_labels)  # 사용자 정의 레이블 적용
axs[1].legend()

# 간격 균등하게 설정 및 여백 제거
plt.savefig("output.png", bbox_inches="tight")
plt.show()
