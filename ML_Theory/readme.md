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


## ✍ 03. 범주형 데이터 처리
범주형 데이터의 경우에는 수치형 데이터로 변환을 해줘야 함
### 방법 1. Label-Encoding

### 방법 2. One-hot Encoding


## ✍ 04. train/test set 분리


## ✍ 05. Feature Engineering

