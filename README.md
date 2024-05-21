![딥러닝_1](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/02e205eb-8bf0-4e3f-ac78-6509e5e6e637)# DeepLearning_Character-Level-Language-Models
DeepLearning Course HW3 - Shakespeare NN

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
- RNN (Training Loss & Validation Loss)
![딥러닝_1](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/d6b40541-402a-4c7e-9f41-e7857b6f1e5a)
- LSTM (Training Loss & Validation Loss)
![딥러닝_2](https://github.com/YewonMin/DeepLearning_Character-Level-Language-Models/assets/108216502/a0285c33-2c69-452a-84bd-91b8a70b7935)


#### 2. Comparison between RNN and LSTM
- LeNet-5
![image](https://github.com/YewonMin/DeepLearning_Classification/assets/108216502/1e8d7a1d-e4ce-4303-8ecf-8353716cade0)
- Custom MLP
![image](https://github.com/YewonMin/DeepLearning_Classification/assets/108216502/aaa67cf7-d61f-4214-a04e-d9ab12ebcbf5)
- 두 모델의 훈련 및 테스트 정확도를 살펴보면, LeNet-5의 train 및 test Acuury가 Custom MLP보다 거의 비슷하거나 항상 높음
- 따라서, 위의 결과를 통해 기본 basic LeNet-5에서는 MNIST 데이터셋에서 Custom MLP보다 더 좋은 예측 성능을 보인다는 것을 알 수 있다.

#### 3. Generate characters with BEST trained model

