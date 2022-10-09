# 용어 정리

## 1. Generalization
![image](https://user-images.githubusercontent.com/102525066/193611578-83710843-1940-40e3-b38b-b1647e2b5ea2.png)

학습 데이터와 훈련데이터의 차이를 Generalizaion gap이라고 부르며,

**Generalizaion이 좋다**는 말은 **학습 데이터와 훈련 데이터의 성능이 비슷하다**라는 뜻

## 2. Underfitting vs Overfitting
![image](https://user-images.githubusercontent.com/102525066/193613744-289fd180-bc3f-493e-a8e0-8af7dede75cd.png)

* Underfitting : 학습 데이터가 적어서 테스트 성능이 떨어지는 것을 의미함(예측을 못함)
* Oerfitting : 학습 데이터가 많아서 학습 정확도는 높지만, 테스트 정확도는 떨어지는 것을 의미함

## 3. cross-validation
![image](https://user-images.githubusercontent.com/102525066/193613031-a1a11aae-798c-46e1-9d82-d98f59263566.png)

* train-data의 일부를 validation data로 만드는 것을 의미함
* k개의 fold로 나눠서 cross-validation을 진행함 (예: 10만개의 train-data, 5개 fold, 2만개씩 돌아가면서 validation data로 활용)
  
※ cross-validation의 목표: hyperparameter값 찾기

### 4. Bias & Variance

### 5. Boost strapping

### 6. Bagging vs Boosting

# 최적화 방법

### Gradient Descent Methods
* 편미분을 이용하여 loss function을 구하는 방법
