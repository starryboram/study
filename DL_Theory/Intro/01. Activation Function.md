# 활성화 함수란?
입력 신호의 총합을 출력 신호로 변환하는 함수를 일컫는다.

[식1] **y = h(a)**

[식2] **a = w1×x1 + w2×x2 + w3×x3 + b**

위 식에서 볼 때 h( )부분을 활성화 함수라고 보면 된다.
활성화 함수는 임계값을 경계로 출력 부분이 바뀐다.(step-funcion을 이용한다)

### Step Function
* 0보다 큰 값은 1로 출력하게끔, 그 외의 값은 0으로 출력하게 만들어주는 함수(비선형함수)

```python
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
```
![image](https://user-images.githubusercontent.com/102525066/194744555-f0d04d51-87a0-4210-ad7c-4ad27ee0f6c8.png)


> 참고: **선형함수 vs 비선형함수**
>
> 직선 1개로 그릴 수 있으면 선형함수, 못 그리면 비선형함수
>
> 딥러닝에서는 비선형함수를 사용해야함(직선 1개로 구분해내면 망이 깊어질 이유가 없음)

## 01. Sigmoid Function
* 0에서 1까지의 값을 가짐(데이터 평균은 0.5)
* 장점: 이진분류에서 마지막 layer로 많이 사용된다.(비선형함수)
* 단점: Vanishing gradient(input값이 작거나 크면 기울기가 아주 작아지면서 발생하는 문제)
* 상세 설명
  - Sigmoid로 여러 layer를 쌓았을 경우에는 0.1 × 0.1 ± .... ->0.0000001 기울기로 나옴
  - gradient가 거의 사라짐 -> 학습이 잘 되지 않음
 
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
![image](https://user-images.githubusercontent.com/102525066/194744717-bca13a0e-7b3f-438d-bc57-36c2e212b2a4.png)

## 02. Tanh Functions
* Sigmoid와 비슷하지만 -1에서 1까지의 값을 가짐(데이터 평균이 0임)
* 장점: 대부분의 경우 sigmoid보다는 성능이 좋음
* 단점: Vanishing gradient 문제 동일 발생

```python
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
```
![image](https://user-images.githubusercontent.com/102525066/194745488-d2f2510c-186b-49e4-8aa4-d22d04bf0780.png)

## 03. ReLU Functions
* sigmoid, tanh 함수보다 최근에 더 많이 쓰는 함수
* 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0이하면 0을 출력하는 함수
* 장점: 기울기가 0이 아니여서 학습이 빠르다.(참고: gradient가 0이 될수록 학습이 느려짐)
* 참고: 대부분의 노드의 값은 0보다 커서 기울기가 0이 되는 경우는 별로 없음

```python
def relu(x):
    return np.maximum(0, x)
```
![image](https://user-images.githubusercontent.com/102525066/194744969-76c646f9-83db-4e53-824f-a076bce2beda.png)

## 04. Leaky ReLU Functions
* 입력 값이 음수일 경우 기울기가 0.01값을 갖게 함
* 장점: ReLU보단 학습이 더 잘된다.

```python
def leaky_relu(x):
    return np.maximum(0.01*x, x)
```
![image](https://user-images.githubusercontent.com/102525066/194745647-df0ad31a-2cde-44c6-98f4-679d3dc43c74.png)

