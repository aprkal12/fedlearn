# 모델 경량화를 통한 RESTful API 기반 동적 연합학습 플랫폼

이 프로젝트는 졸업연구로 진행한 연합학습 플랫폼입니다.  
RESTful API를 활용하여 동적 연합학습 플랫폼을 설계하고, 모델 경량화를 통해 파라미터 데이터 전송 효율을 높이며 대시보드를 활용한 실시간 모니터링 및 제어가 가능합니다.

---

## 📚 연구 배경
- 딥러닝 기술의 발전으로 다양한 분야에서 연구가 활발히 진행되고 있으나, 데이터 중앙화에 따른 개인정보 보안 문제가 이슈.
- 이를 해결하기 위해 연합학습(Federated Learning)이 제안됨.
  - 데이터 중앙화 없이 각 로컬에서 학습 후 파라미터를 중앙으로 전송.
  - 전송된 파라미터를 집계하여 글로벌 모델을 업데이트.
- 기존의 연합학습 선행연구와 플랫폼들의 한계
  - gRPC 통신 기반의 기존 연합학습 프레임워크들은 연합학습 초기 구성을 변경하거나 학습 중 클라이언트 환경변화에 즉각적인 대처가 어려움

---

## 🎯 연구 목표
1. **RESTful API 기반 동적 연합학습 구현:**
   - 클라이언트의 자유로운 참여와 이탈 지원.
2. **모델 경량화를 통한 전송 효율 개선:**
   - 파라미터 raw data를 경량화하고 데이터 전송에 JSON 대신 직렬화 및 압축 방식을 활용.
3. **대시보드 설계:**
   - 실시간 학습 상태 및 클라이언트 관리 기능 제공.

---

## 🔍 제안 방안
### 플랫폼 구조
- HTTP URI 설계를 통한 연합학습 기능 명세.
- REST 기반 통신을 사용하되, 대용량 데이터 전송 효율을 위해 **모델 경량화** 적용:
  - Float32 → Bfloat16 변환.
  - Pickle 모듈을 활용한 직렬화 및 **zstd 압축 알고리즘** 적용.

### 아키텍처

![architecture](https://github.com/aprkal12/fedlearn/blob/master/static/architecture.jpg?raw=true)

### URI 기능 명세

![URI](https://github.com/aprkal12/fedlearn/blob/master/static/uri.jpg?raw=true)

### 대시보드
<img src="https://github.com/aprkal12/fedlearn/blob/master/static/dashboard2.png?raw=true" width="600" height="500"/>

<img src="https://github.com/aprkal12/fedlearn/blob/master/static/dashboard.png?raw=true" width="500" height="600"/>

- **클라이언트 모니터링:**
  - 상태: `join`, `ready`, `training`, `update`, `finish`.
- **글로벌 모델 모니터링:**
  - 정확도, 참여 클라이언트 수, 라운드 상태 등 실시간 확인.

---

## 🧪 실험 및 구현 결과
- ### 실험 환경:
  - 서버와 다수의 클라이언트 구성.
  - 모델: ResNet-18, ResNet-50.
  - 데이터셋: CIFAR-10 / 6만장의 이미지 데이터 / IID, Non-IID 데이터 환경 구성.

    **IID / Non-IID 데이터 구성**

    <img src="https://github.com/aprkal12/fedlearn/blob/master/static/IID.png?raw=true" width="300" height="300"/> <img src="https://github.com/aprkal12/fedlearn/blob/master/static/NonIID.png?raw=true" width="300" height="300"/>

- ### 실험 결과:
  - IID 환경: 클라이언트 수 증가 시 정확도는 약간 감소.
  - Non-IID 환경: 불균형 데이터에서 안정적인 정확도 유지.
  - 모델 경량화를 통해 JSON 대비 약 **93% 용량 감소**.
  - Zstd 알고리즘 적용으로 압축 시간과 효율 간의 최적화 달성.

    **모델 경량화**
    - JSON, Binary, Raw data, 제안 방안 비교
    - JSON 대비 약 93%, 직렬화 및 원본데이터 대비 약 60% 용량 감소

    <img src="https://github.com/aprkal12/fedlearn/blob/master/static/result.png?raw=true" width="500" height="300"/>

    **제안 플랫폼을 이용한 연합학습 수행 결과**
    - **중앙집중식 학습 방법(기존의 딥러닝 방식)을 통해 100회 학습하여 얻는 정확도와 비교**
      - 연합학습을 통해 중앙집중식 학습에 근접하는 모델 정확도 확보 
      - 데이터 수가 고정된 실험환경으로 클라이언트 수가 많아질 수록 클라이언트 로컬 데이터에 과적합되어 모델 집계 후 성능이 저하되는 것으로 추정됨
  
    ![graph](https://github.com/aprkal12/fedlearn/blob/master/static/output.png?raw=true)
    
---

## 🛠 기술 스택
- **Backend:** Python, Flask, web socket
- **Frontend:** HTML/CSS, JavaScript
- **Framework:** PyTorch

---

## 👥
| Name  | Contact               |
|-------|-----------------------|
| 김동현 | aprkal12@naver.com    |

---

## 📢 결론
- RESTful API를 통해 동적인 연합학습을 구현하고 대시보드를 통해 학습 진행 및 관리 기능 제공.
- 모델 경량화를 통해 데이터 전송 효율을 높이고 클라이언트 참여 유연성을 확보.
- 한계 : 다양한 모델과 다양한 클라이언트(모바일, 엣지 디바이스 등)와의 호환성, 모델 자료형 변경으로 인한 정밀도 감소
- 향후 연구: 다양한 클라이언트 환경에서도 성능 최적화 연구 진행.


