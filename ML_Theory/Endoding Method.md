# 1. Label Encoder
* 카테고리형 피처를 숫자로 변환시키는 방식
* 하는 이유: 단어, 글자를 숫자로 바꿔줘야 학습이 가능함
* 변환시키는 방법: 알파벳순서, 가나다라 순서대로 적용됨
* 순서에 의미가 있을 때(유치원, 초등학교, 직급 등), 고유값의 갯수가 많을 때 사용하기

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
* 이러한 특성 때문에 Label encoding은 선형회귀같은 머신러닝 알고리즘에는 적용 X 
* 트리 계열의 머신러닝 알고리즘은 숫자의 이러한 특성을 반영하지 않기 때문에 문제 없음

# 2. One-Hot Encoder
* label encoding이 된 후에 loss값을 줄이기 위해 하는 작업
* 해당 고유값에만 1로 표현하는 방식(나머지 컬럼에는 0으로 표시)
* 순서에 의미가 없을 때(국가명, 회사명 등), 고유값의 개수가 많지 않을 때(많으면 메모리소비가 많아서 비효율적) 사용하기
* 
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

# 3. Ordinal Encoder
* 순서형 자료에 적합한 인코딩 방식(Label encoding 방식이랑 비슷)
* 알파벳 순서까지 고려 가능하며 Pandas를 사용하여 인코딩 가능
* 하지만, 사용자가 주어진 범주형 변수를 보고 직접 변수 값들 간의 순서(order)를 dictionary 형태로 정의해줘야 함. 
* 그래서 직관적이긴 하지만 추가적인 코딩을 해야하는 수고가 필요하다.

# 4. Helmert Encoder
* 특정 범주형 변수에서 특정한 수준의 인코딩 값을 도출해내기 위해서 해당 level에 매핑되는 종속변수(즉, y값)의 평균값과 모든 level에 매핑되는 모든 종속변수값들의 평균값을 비교하는 방법 
* 인코딩하려는 범주형 변수는 Temperature이고 종속변수는 Target임.(아래 그림 참조)
```python
import category_encoders as ce
encoder = ce.HelmertEncoder(cols=['Temperature'], drop_invariant=True)
dfh = encoder.fit_transform(df['Temperature'])
df = pd.concat([df, dfh], axis=1)
df
```
![image](https://user-images.githubusercontent.com/102525066/193507466-74918860-5963-440f-b823-a782285faf23.png)


# 5. Frequency Encoder
* 빈도수를 토대로 라벨로 활용하는 방법
* 빈도수가 타겟을 예측하는데 관련이 있을 경우, 모델의 성질에 따라 비례율, 반비례율 정도를 인지하고 지정
* 방법: 변환하고자 하는 범주형특성 선택 하여 group by해서 빈도수를 세고, 훈련 데이터에 더해주는 방식

```python
fe = df.groupby("Temperature").size()/len(df)
df.loc[:, "Temp_freq_encode"] = df["Temperature"].map(fe)
df
```
![image](https://user-images.githubusercontent.com/102525066/193507350-c54a9e07-c9b4-48a1-9067-c518d6597761.png)

# 6. M-Estimate Encoder
* target encoding 의 단순화된 버전(과적합 단점 극복을 위해 개발됨)
* 정규화 정도를 나타내는 m 이 가장 중요한 하이퍼파라미터이며, m값이 높을 수록 수축되는 형태를 강하게 보임
* m값은 1~100사이의 값으로 할 것을 추천함
![image](https://user-images.githubusercontent.com/102525066/193506371-8b0f742c-df7f-4dea-ad6f-916091a69e6f.png)

```python
%%time
MEE_encoder = MEstimateEncoder()
train_mee = MEE_encoder.fit_transform(train[feature_list], target)
test_mee = MEE_encoder.transform(test[feature_list])
```

# 7. Weight Of Evidence Encoder

### 신용 및 금융 산업에서 대출 불이행 위험 예측 모델로 활용하기 위해 개발됨
* 신용평가에서 일반적으로 사용되는 목표 기반 인코더 (증거 가중치)
* 좋은 위험과 나쁜 위험을 구분하기 위해 강도를 측정함.
* WOE는 증거가 가설을 지지하거나 약화시키는 정도를 측정
* (예) 사람 A가 상환 능력이 40%라고 가정했을 때, 대출 불이행 위험 정도는 ??? 정도다. (??? ⇒ 가중치를 두어 계산함) 
* 단점: target leakage 및 과적합으로 이어질 수 있음.

```python
a = Distribution of Good Credit Outcomes
b = Distribution of Bad Credit Outcomes
WoE = ln(a / b)
```

```python
%%time
WOE_encoder = WOEEncoder()
train_woe = WOE_encoder.fit_transform(train[feature_list], target)
test_woe = WOE_encoder.transform(test[feature_list])
```

# 8. James-Stein Encoder
* Target 기반 encoder방식. 
![image](https://user-images.githubusercontent.com/102525066/193505897-a1aa7029-ebfb-42df-bc74-0c6e94acb828.png)

* B를 선택하는 것이 가장 중요하며, 교차 검증을 통해 하이퍼파라미터 조정
* 단점: 정규 분포에 대하여 적용이 가능함.  
* 이를 피하기 위해 WoE 인코더(간단하기 때문에 기본적으로 사용됨)에서 수행된 log-odds 비율을 이용하여 binary target으로 변환하거나 beta distribution 사용.

```python
%%time
JSE_encoder = JamesSteinEncoder()
train_jse = JSE_encoder.fit_transform(train[feature_list], target)
test_jse = JSE_encoder.transform(test[feature_list])
```

# 9. Leave-one-out Encoder

### 범주형 변수로 이루어진 데이터 셋에서 사용하기

* target encoding과 유사하지만, 이상치를 줄이기 위해 하는 방식으로 해당 row의 target값을 제외한 나머지 평균값을 통해 feature 값을 예상하는 방식.
* 장점: cross-fold 방식보다 빠름
* 단점: XGBoost, CatBoost, LightGBM과 같은 트리 기반 부스팅 알고리즘에 도움이 안됨. target leakeage 부분도 단점

```python
LOOE_encoder = LeaveOneOutEncoder()
train_looe = LOOE_encoder.fit_transform(train[feature_list], target)
test_looe = LOOE_encoder.transform(test[feature_list])
```

# 10. Catboost Encoder

### 범주형 변수로 이루어진 데이터 셋에서 사용하기 (예측 성능 우수_최근 모델)

* 특성끼리 조합이 항상 정해져 있을 때 사용하는 방법. (한 변수만 사용해서 택하는 방법)
* 항상 한국은 삼성, 미국은 애플처럼 조합이 일정한 경우에는 Overfitting을 방지하기 위해서 변수 하나만 선택하여 encoding 하는 방식 
  →  이 방식을 **Categorical feature combination**이라 부름
* 참고로, 데이터 대부분이 수치형 변수인 경우, Light GBM보다 학습 속도가 느림.
* 장점 1: one-hot encoding, label encoding 작업을 하지 않아도 그대로 모델의 input 사용 가능(단, 범주형 변수의 cardinalitry가 작은 경우에는 one-hot-encoding 진행)
* 장점 2: 기본 파라미터가 최적화가 잘 되어있어서 파라미터 튜닝에 크게 신경 쓰지 않아도 됨 (대부분 부스팅 모델이 파라미터 튜닝을 하는 이유는 트리의 다형성과 오버피팅의 문제를 해결하기 위함인데, catboost 같은 경우에는 내부적인 알고리즘으로 해결하고 있어서 굳이 파라미터 튜닝이 필요없음)

### Ordered Target Encoding
* 일반 target encoding 방식을 사용하게 되면 target 값의 평균으로 대체됨. → data leakage 문제 발생 → Catboost의 경우에는 Ordered Target Encoding 사용
*  Ordered Target Encoding 의 경우 과거의 데이터를 이용해 현재의 데이터를 인코딩함.

### Ordered Boosting
* train set의 순서를 랜덤으로 섞음(Random Permutation 진행)
* 이 때, permutation으로 생성할 데이터셋 갯수는 4개(Defalt) → 지정 가능
* Random Permutation을 하면 랜덤하게 순서를 섞기 때문에 Overfitting 방지 가능

```python
%%time
CBE_encoder = CatBoostEncoder()
train_cbe = CBE_encoder.fit_transform(train[feature_list], target)
test_cbe = CBE_encoder.transform(test[feature_list])
```

[참고 사이트]

https://techblog-history-younghunjo1.tistory.com/99

https://conanmoon.medium.com/%EB%8D%B0%EC%9D%B4%ED%84%B0%EA%B3%BC%ED%95%99-%EC%9C%A0%EB%A7%9D%EC%A3%BC%EC%9D%98-%EB%A7%A4%EC%9D%BC-%EA%B8%80%EC%93%B0%EA%B8%B0-%EC%9D%BC%EA%B3%B1%EB%B2%88%EC%A7%B8-%EC%9D%BC%EC%9A%94%EC%9D%BC-7a40e7de39d4

[https://www.linkedin.com/pulse/encode-categorical-features-revanth-yadama](https://www.linkedin.com/pulse/encode-categorical-features-revanth-yadama)

[https://brendanhasz.github.io/2019/03/04/target-encoding#leave-one-out-target-encoding](https://brendanhasz.github.io/2019/03/04/target-encoding#leave-one-out-target-encoding)

[https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8](https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8)

[https://www.kaggle.com/code/subinium/11-categorical-encoders-and-benchmark](https://www.kaggle.com/code/subinium/11-categorical-encoders-and-benchmark)

[https://hyewon328.tistory.com/entry/CatBoost-CatBoost-알고리즘에-대한-이해](https://hyewon328.tistory.com/entry/CatBoost-CatBoost-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4)
