# DeepLearning Course HW3 - Shakespeare NN

## Folder Structure
<<<<<<< HEAD
```bash
├── dataset.py: Shakespeare 데이터셋을 로드하고 전처리하는 코드가 포함되어 있습니다.
├── model.py: Vanilla RNN & LSTM 모델을 정의하는 코드가 포함되어 있습니다.
├── main.py: 모델을 훈련하고 평가하며, train과 validation의 loss를 플롯하는 메인 코드가 포함되어 있습니다.
├── *generate.py: 가장 좋은 검증 성능을 보이는 모델을 선택하여 학습된 모델로 문자를 생성합니다. 
```

## 0. Plot the average loss values for training and validation
##### RNN (Training Loss & Validation Loss)
![딥러닝_1](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/8ab9aa10-df33-44d2-a44c-c6f5e24cb224)
* training, validation 모두 6 epoch에서 가장 낮은 loss값을 보이고 있지만, 그 후 소폭 상승함
##### LSTM (Training Loss & Validation Loss)
![딥러닝_2](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/cd8e44ec-aea2-4a53-aa9a-303de7897576)
* training, validation 모두 20 epoch까지 안정적으로 loss값이 감소하는 것을 볼 수 있음

## 1. Comparison between RNN and LSTM
#### 1-1. 총 20 epoch에 대한 RNN, LSTM의 전체 train & validation loss 결과
![image](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/4ab95fef-8b51-43dc-b8c4-d4136db63fd8)
#### RNN
- Training Loss: 초기 손실값은 1.5379에서 시작하여 1.3978로 약간 감소함
- Validation Loss: 초기 손실값은 1.4254에서 시작하여 1.4003로 약간 감소함
#### LSTM
- Training Loss: 초기 손실값은 1.5019에서 시작하여 0.9555로 상당히 감소함
- Validation Loss: 초기 손실값은 1.3097에서 시작하여 0.9848로 상당히 감소함

#### 1-2. 비교 분석
##### 훈련 손실(Training Loss):
- RNN: 훈련 손실값이 전반적으로 줄어들지만, 일정 epoch 이후 다시 상승하며 감소폭이 크지 않음
- LSTM: 훈련 손실값이 매우 빠르게 줄어들며, 최종적으로 RNN보다 훨씬 낮은 값을 가짐

##### 검증 손실(Validation Loss):
- RNN: 초기에는 감소하지만 이후 다시 증가하는 경향을 보임. 이는 과적합(overfitting)의 징후일 수 있음
- LSTM: 지속적으로 감소하는 경향을 보이며, 최종적으로 RNN보다 낮은 손실값을 가짐. 이는 LSTM 모델이 RNN보다 더 일반화(generalization) 능력이 뛰어남을 의미함

#### 1-3. 결론
- LSTM의 성능이 더 우수함: 검증 손실값이 더 낮고, 훈련 손실값 또한 더 낮음. 이는 LSTM이 RNN에 비해 더 나은 언어 생성 성능을 가짐을 시사함
- RNN의 과적합 가능성: 검증 손실값이 감소한 후 다시 증가하는 양상은 훈련 데이터에 과적합되어 새로운 데이터에 대한 성능이 떨어질 수 있음을 나타냄

## 2. Generate characters with BEST trained model
BEST trained model: LSTM
#### 2-1. different seed characters: ['S', 'T', 'O', 'A', 'G']
#### 2-2. Softmax 함수 with temperature T
다양한 온도 값(T=0.5, 1.0, 1.5)으로 텍스트를 생성하고, 결과를 분석하여 온도 값이 생성된 텍스트에 미치는 영향 파악
* T = 1: 기본 설정으로, 모델의 예측 확률 분포를 그대로 사용
* T < 1: 분포를 더 날카롭게 만들어, 더 확실한 예측을 하게 함 (예: T = 0.5)
* T > 1: 분포를 부드럽게 만들어, 덜 확실한 예측을 허용 (예: T = 1.5)
#### 2-3. 결과
##### Temperature: 0.5
```bash
Seed Character: S | Generated Text: S:
Marcius,
He is with your house, my lord, as we have a spacle! the earth have put unto the whole st
```
```bash
Seed Character: T | Generated Text: There is the one the death of your swords
Our angry grandam to my soul, I pray you, and wife, and you
```
```bash
Seed Character: O | Generated Text: ORIOLANUS:
Shall I be put upon 't; and the good of men
Than when I see thee thy hearts sorrown to the
```
```bash
Seed Character: A | Generated Text: ANUS:
The senate crown'd: your lordship at the royal duke more from fellow
The soul for the leads and
```
```bash
Seed Character: G | Generated Text: GLOUCESTER:
What then let the people's power to be so diverse my curses!
I ravid conditions, which th
```
##### 결과분석:
* 생성된 텍스트가 문법적으로 더 일관되고, 의미 있는 문장이 많은 것으로 보임
* 예측이 모델의 확실한 결정에 의해 주도되기 때문에, 생성된 텍스트가 더 자연스럽고 논리적인 흐름을 가짐
* 그러나 예측의 다양성이 낮아 반복적이거나 예상 가능한 패턴이 나타날 수 있음
##
##### Temperature: 1.0
```bash
Seed Character: S | Generated Text: S:
I'll walls, as more voice, and you should heard
As I entery men that I have not jet you o'erlood
W
```
```bash
Seed Character: T | Generated Text: TH:
Back out, do not stand upon enge?

LADY ANNE:
Artful three off! thou live to our service,
Is not 
```
```bash
Seed Character: O | Generated Text: OLANUS:
What, my gracious lord!

First Servingman:
O, believe us heart!

SICINIUS:
Why, then, then,
F
```
```bash
Seed Character: A | Generated Text: A
My holy blind
In all our wall'd by the invejook of are less deared by be necess'd heart's curse.

B
```
```bash
Seed Character: G | Generated Text: GLOUCESTER:
The gods have such being answer'd up:
The man in the council undone army, and her heel
An
```
##### 결과분석:
* 생성된 텍스트가 더 다양한 표현을 포함하고 있음
* 낮은 temperature 값보다 더 창의적이고 예측할 수 없는 텍스트를 생성함
* 문법적으로 다소 불안정한 부분이 있을 수 있지만, 전반적으로 더 풍부한 언어 표현을 포함하고 있음
##
##### Temperature: 1.5
```bash
Seed Character: S | Generated Text: S:
I know, that prithee; morrow here began's inclicamed tency,
Muaked-cond Savookts by't again.

GLOU
```
```bash
Seed Character: T | Generated Text: Tower: I not knee heavy; it's good?
O Good throad, not, well-deidst thy charge
And, which we would he
```
```bash
Seed Character: O | Generated Text: OMenes of Towerds fight. Hads IItitud--
Hislous' light,
From weeds got I common
Of nore I faulmily im
```
```bash
Seed Character: A | Generated Text: ALERIA:
No nests great one
That stock i' the nobility.

Th, Alif the afpace: the man, so loud in fail
```
```bash
Seed Character: G | Generated Text: GLOLUNipising: whether's patience.
Officiaten we haply false your wife of Pomfret him. Althis war mad
```
##### 결과분석:
* 생성된 텍스트가 매우 다양하고 예측할 수 없는 결과를 포함하고 있음
* 높은 temperature 값은 모델이 덜 확실한 결정을 내리게 하여 창의적인 텍스트를 생성하지만, 문법적으로 일관성이 떨어질 수 있음
* 결과적으로 텍스트가 더 무작위적으로 보일 수 있음

##
#### 2-4. 결론
* Temperature: 텍스트 생성에서 모델의 샘플링 확률 분포의 스케일을 조절하는 역할 (낮은 temperature: 모델의 예측을 더 확실하게, 높은 temperature: 더 다양하지만 덜 확실한 예측을 만듦) 
* 결론적으로, 텍스트 생성의 목적에 따라 temperature 값을 선택하는 것이 중요함. 더 일관된 결과를 원한다면 낮은 temperature (ex., 0.5)를 사용하고, 더 창의적이고 다양한 결과를 원한다면 높은 temperature (ex., 1.0 or 1.5)를 사용할 수 있음
* 따라서, 모델이 생성하는 텍스트의 품질과 다양성 사이에서 적절한 균형을 찾는 것이 중요하다고 판단됨
