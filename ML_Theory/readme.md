## ✍ 00. 데이터 전처리란?
* 머신러닝 적용에 있어 가장 중요한 부분은 '데이터 전처리'
* 일반적으로 데이터 분석가는 모델 성능 향상을 위해 70% 이상의 시간을 전처리에 할애
* 결측값 처리, 정규화/표준화 작업, 범주형 데이터 처리, train/test set 분리, Feature Engineering 등을 수행

## ✍ 01. 결측값 처리
### 1) 결측값 확인
> 방법 1 : **data.info()**  : data의 전체 information을 보여주며 type과 non-null 값 확인에 유용
>
> 방법 2 : **data.isnull().sum()** : null값을 다 합산해서 알려줌
    
### 2) 결측값 일반적 처리
null값을 처리할 때에는 데이터 대비 얼마나 null값이 차지하는 지에 따라 방법이 달라짐
* 10% 미만일 경우 → 해당 row값을 삭제하거나 다른 값으로 대체해줌
* 10~50% 미만일 경우 → 모델 만들어서 처리
* 50% 이상일 경우 → 해당 column 자체를 삭제하는 것이 나을 수 있음

### 3) 결측값 삭제하기
* column 삭제하기

  > 방법 1: data_2 = data.drop(columns='weather', axis = 1, inplace = False)
  > 
  > 방법 2: data.drop(['Weather'], axis=1, inplace=True)

* row 삭제하기

  > 방법 1: data.dropna()
  > 
  > 방법 2: data.dropna(axis=0)

### 4) 결측값 대체하기
대체 방법 | 코드 
---|---
0으로 대체 |data.fillna(0)
중간값으로 대체 | data['Fee'].fillna(data['Fee'].median(), inplace = True)
평균값으로 대체하기 | data.fillna({'Fee':data['Fee'].mean()})
결측값의 앞 행 값으로 대체| data.fillna(method='ffill') 혹은 data.fillna(method='pad')
결측값의 뒷 행 값으로 대체| data.fillna(method='bfill') 혹은 data.fillna(method='backfill')
특정값으로 결측치 대체| data.fillna('특정값')
## ✍ 02. 정규화/표준화 작업
* 수치형 데이터 전처리에서 사용되는 요소
* 작업 이유: 학습 데이터의 feature 간 단위나 스케일(scale) 차이가 클 수록 학습 모델의 성능이 저하될 수 있음. 
            따라서 데이터의 단위를 맞춰주는 작업이 필요하며, 이를 scaling이라고 말함.

### 정규화(Normalization): Min-Max scaler 방식

    a. 특징: 0~1 사이로 수치를 변화시켜 주며, 구간을 정해줘야함
    b. 장점: 모델 학습의 성능이 좋아지고, 학습 시간이 줄어드는 효과가 있음. Overflow 방지 효과도 있음
    c. 단점: 이상치(Outlier)에 많은 영향을 받음
    d. 비고: 수치형컬럼만 정규화가 가능함. 이상치 단점을 보완하기 위해서는 Z-score normalizaion을 사용해봐도 좋음.
    
### 표준화(Standardization) : StandardScaler

    a. 특징: 평균이 0이고, 표준편차가 1의 분포로 만들어줌. 비지도 학습(최대 최소의 범위를 알기 힘든 상황)에서 사용
    b. 비고: Feature간 단위 차이가 극심할 경우에는 정규화보다는 표준화를 사용하는게 좋음  
            정규화의 경우에는 0 ~ 1범위로 압축하기 때문에 정보 손실이 발생할 우려가 있음

## ✍ 03. 범주형 데이터 처리
범주형 데이터의 경우에는 수치형 데이터로 변환을 해줘야 함
### 1. Label-Encoding
* 카테고리형 피처를 숫자로 변환시키는 방식
* 하는 이유: 단어, 글자를 숫자로 바꿔줘야 학습이 가능함
* 변환시키는 방법: 알파벳순서, 가나다라 순서대로 적용됨
* 순서에 의미가 있을 때(유치원, 초등학교, 직급 등), 고유값의 개수가 많을 때 사용하기

```python
from sklearn.preprocessing import LabelEncoder

items = ['김밥', '떡볶이', '새우튀김', '김말이튀김', '김치치즈볶음밥', '김밥']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값', labels) # 결과값: 인코딩 변환값 [1 3 4 0 2 1]
print('인코딩 클래스', encoder.classes_) 
# 결과값: 인코딩 클래스 ['김말이튀김' '김밥' '김치치즈볶음밥' '떡볶이' '새우튀김']
print('디코딩 원본값', encoder.inverse_transform([1 3 4 0 2 1]))
# 예측 결과에 적용할 수 있도록 역변환이 필요함
```

**주의할 점**
* 숫자로 변환 시 큰 값에 가중치를 부여하거나, 큰 값을 중요하게 인식할 가능성이 있음
* 이러한 특성 때문에 Label encoding은 선형회귀같은 머신러닝 알고리즘에는 적용 안 
* 트리 계열의 머신러닝 알고리즘은 숫자의 이러한 특성을 반영하지 않기 때문에 문제 없음

### 2. One-hot Encoding
* label encoding이 된 후에 loss값을 줄이기 위해 하는 작업
* 해당 고유값에만 1로 표현하는 방식(나머지 컬럼에는 0으로 표시)
* 순서에 의미가 없을 때(국가명, 회사명 등), 고유값의 개수가 많지 않을 때 사용할 것(고유값의 개수가 많으면 메모리 소비가 많아 비효율적)

```python
# 방법 1)
from sklearn.preprocessing import OneHotEncoder
import numpy as np

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '믹서', '믹서']
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
labels = labels.reshape(-1,1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print(oh_labels.toarray())
print(oh_labels.shape)

# 방법 2)
import pandas as pd
df = pd.DataFrame({'item': ['김밥', '떡볶이', '새우튀김', '김말이튀김', '김치볶음밥']})
pd.get_dummies(df)
```

이 외에도 여러 encoding 방식이 있음. 더 많은 정보를 보고 싶을 경우 
## ✍ 04. train/test set 분리


## ✍ 05. Feature Engineering

