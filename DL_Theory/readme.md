# Deep Learning
### 1. 딥러닝의 요즘
예전에는 "사진에 '강아지', '사람', '자동차' 가 있다."정도 수준이었다면,

요즈음 딥러닝을 학습시키는 건 '사람이 강아지와 함께 자동차를 탄다'를 표현하게하는 **이해**하게 하는 데에 목적을 두고 있다.

즉, 수식을 이용해서 학습하는 방식이었다면(강아지의 코, 눈 위치 등을 수식적으로 표현) **경험/패턴 기반으로 학습**하는 방식이다.

### 2. 딥러닝의 개요
![image](https://user-images.githubusercontent.com/102525066/193408869-14bd0273-a621-4fe1-af65-3d82236a8fd0.png)

입력(input)에 중요성(Weight)을 매겨서 학습하게 하는 과정이며, 이를 **퍼셉트론(Perceptron)** 이라고 한다.

# Deep Learning 종류
### 1. ANN ( Artificial Neural Network ) : 인공 신경망
- 3개 이상의 layer를 가지고 있음
- 단점: 학습시간이 느림, Overfitting 발생, 적절한 파라미터 값 찾기 어려움

### 2. DNN (Deep Neural Network)  : 심층 신경망
- 은닉층을 2개 이상
- ANN보완모델로, 많은 데이터와 반복학습, 사전 학습과 오류역전파 기법을 통해 널리 사용됨
![image](https://user-images.githubusercontent.com/102525066/194695559-8e15a21f-ba21-4f46-a072-94a3366eecd0.png)

△ ANN / DNN 비교 그림
 
### 3. CNN ( Convolution Neural Network) : 합성곱 신경망
- DNN을 응용한 알고리즘
- 데이터의 특징을 추출 후 패턴 파악(input image에 Convolution kernel을 적용하여 feature map 생성
- Convolution(특징 추출), Pooling(layer size 줄여줌) 과정이 들어감
- 종류: AlexNet, VGGNet, GoogleNet, ResNet 등 
- 사용 예) 알파고
![image](https://user-images.githubusercontent.com/102525066/194695521-81977070-d2f0-4dd3-8efb-99de6eac417c.png)

### 4. RNN ( Recurrent Neural Network) : 순환 신경망
- 반복적이고 순차적인 데이터(Sequential data) 학습에 특화된 인공신경망의 한 종류
- 내부에 순환구조가 포함되어 자연어 처리 분야에서 성능이 좋음
- 현재의 학습과 과거의 학습 연결을 가능하게 하며 시간에 종속된다는 특징이 있다
- 대표: LSTM(Long Short Term memory) 모형
- 사용 예) 파파고, 주식 예측 등
 
### 5. GAN (Generative Adversarial Network) : 생산적 적대 신경망
- 서로 경쟁하면서 가짜 이미지와 진짜 이미지를 최대한 비슷하게 만들어 내도록 하는 신경망 (생성모델)
![image](https://user-images.githubusercontent.com/102525066/194695781-2fe3a041-b7ea-4d8e-be67-4d76817333b2.png)


###### 공부 참고 사이트: https://www.youtube.com/watch?v=qCJL3LD6I1g
https://upgrade-j.tistory.com/entry/python%EB%94%A5%EB%9F%AC%EB%8B%9D-ANN-DNN-CNN-RNN-GAN-%EC%A0%95%EB%A6%AC
