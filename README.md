# DeepLearning Course HW3 - Shakespeare NN

## Folder Structure
<<<<<<< HEAD
```bash
├── dataset.py: Shakespeare 데이터셋을 로드하고 전처리하는 코드가 포함되어 있습니다.
├── model.py: Vanilla RNN & LSTM 모델을 정의하는 코드가 포함되어 있습니다.
├── main.py: 모델을 훈련하고 평가하며, train과 validation의 loss를 플롯하는 메인 코드가 포함되어 있습니다.
├── *generate.py: 가장 좋은 검증 성능을 보이는 모델을 선택하여 학습된 모델로 문자를 생성합니다. 
```

## Result (Report)
#### 0. Plot the average loss values for training and validation
##### RNN (Training Loss & Validation Loss)
![딥러닝_1](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/8ab9aa10-df33-44d2-a44c-c6f5e24cb224)
* training, validation 모두 6 epoch에서 가장 낮은 loss값을 보이고 있지만, 그 후 소폭 상승함
##### LSTM (Training Loss & Validation Loss)
![딥러닝_2](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/cd8e44ec-aea2-4a53-aa9a-303de7897576)
* training, validation 모두 20 epoch까지 안정적으로 loss값이 감소하는 것을 볼 수 있음

#### 1. Comparison between RNN and LSTM
##### 1-1. 총 20 epoch에 대한 RNN, LSTM의 전체 train & validation loss 결과
![image](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/4ab95fef-8b51-43dc-b8c4-d4136db63fd8)
#### RNN
- Training Loss: 초기 손실값은 1.5379에서 시작하여 1.3978로 약간 감소함.
- Validation Loss: 초기 손실값은 1.4254에서 시작하여 1.4003로 약간 감소함.
#### LSTM
- Training Loss: 초기 손실값은 1.5019에서 시작하여 0.9555로 상당히 감소함.
- Validation Loss: 초기 손실값은 1.3097에서 시작하여 0.9848로 상당히 감소함.

##### 1-2. 비교 분석
##### 훈련 손실(Training Loss):
- RNN: 훈련 손실값이 전반적으로 줄어들지만, 일정 epoch 이후 다시 상승하며 감소폭이 크지 않음.
- LSTM: 훈련 손실값이 매우 빠르게 줄어들며, 최종적으로 RNN보다 훨씬 낮은 값을 가짐.

##### 검증 손실(Validation Loss):
- RNN: 초기에는 감소하지만 이후 다시 증가하는 경향을 보임. 이는 과적합(overfitting)의 징후일 수 있음.
- LSTM: 지속적으로 감소하는 경향을 보이며, 최종적으로 RNN보다 낮은 손실값을 가짐. 이는 LSTM 모델이 RNN보다 더 일반화(generalization) 능력이 뛰어남을 의미함.

##### 1-3. 결론
LSTM의 성능이 더 우수함: 검증 손실값이 더 낮고, 훈련 손실값 또한 더 낮음. 이는 LSTM이 RNN에 비해 더 나은 언어 생성 성능을 가짐을 시사함.
RNN의 과적합 가능성: 검증 손실값이 감소한 후 다시 증가하는 양상은 훈련 데이터에 과적합되어 새로운 데이터에 대한 성능이 떨어질 수 있음을 나타냄.

#### 3. Generate characters with BEST trained model
* BEST trained model: LSTM
##### 3-1. different seed characters: ['S', 'T', 'O', 'A', 'G']
##### 3-2. Softmax 함수 with temperature T
다양한 온도 값(T=0.5, 1.0, 1.5)으로 텍스트를 생성하고, 결과를 분석하여 온도 값이 생성된 텍스트에 미치는 영향 파악
* T = 1: 기본 설정으로, 모델의 예측 확률 분포를 그대로 사용합니다.
* T < 1: 분포를 더 날카롭게 만들어, 더 확실한 예측을 하게 합니다. (예: T = 0.5)
* T > 1: 분포를 부드럽게 만들어, 덜 확실한 예측을 허용합니다. (예: T = 1.5)
##### 3-3. 결론
* Temperature: 0.5
Seed Character: S | Generated Text: S:
Marcius,
He is with your house, my lord, as we have a spacle! the earth have put unto the whole st

Seed Character: T | Generated Text: There is the one the death of your swords
Our angry grandam to my soul, I pray you, and wife, and you


