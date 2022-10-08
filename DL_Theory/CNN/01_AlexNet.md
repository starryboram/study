# 개요
* 2012년 우승한 모델
* Input에서 GPU를 2개를 병렬로 사용함
* 5개의 Convolutional layer + 3개의 Fully connected layer로 이루어져 있음
* Output은 1000가지의 class로 도출됨

# 논문 읽기
* 1000개의 이미지 카테고리 사용 (Test-set이 라벨링 된 ILSVRC-2010 데이터 주로 사용)
* 120만개의 Train-set, 5만개의 validation set, 15만개의 test set 사용
* 모든 이미지를 256×256 사이즈의 RGB 컬러 사진으로 크기를 통일함
* 직사각형 사진의 경우, 길이가 짧은 곳에 맞춰 256으로 rescale후 가운데 중심으로 자름

# 방법
### 1. Activation layer: ReLU 

### 2. GPU 2개 사용
### 3. Normalization: Local response
### 4. Overlapping pooling 방식 도입
### 5. Data augmentation, Drop-out 방식 사용




##### 출처 https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
