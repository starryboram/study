# 🧨 Semantic Segmentation
* 의미: 이미지 픽셀마다 분류하는 방법 -> 픽셀의 라벨이 어떤걸 의미하는지를 알려줌
* 다른 용어: Dense classification, Per Pixel classification이라고도 불림
* 활용 분야: 자율주행 분야(인도, 사람, 차도, 신호등 등을 구분해야 하기 때문)

## 🧨 FCN(Fully Convolutional Network)
![image](https://user-images.githubusercontent.com/102525066/194119497-b736498b-bace-4eb9-8bbd-6f0e0a8c607a.png)

* 왼쪽 그림에서 Dense layer를 없앤 것이 Fully Convolutional Network
* 두 방식 모두 parameter가 똑같은데, **FCN을 쓰는 이유는? input image 사이즈에 구애받지 않는다.**
>
> Dense layer를 쓰는 경우: input image size(spatial demension)가 크면 output size도 비례해서 커짐(heatmap같은 효과)
> 
> Dense layer를 안 쓸 경우: 100×100 input image가 10×10로 줄어들기 때문에 input size에 영향을 받지 않음
> 
> **※ 주의할 점: 사이즈가 줄어들었기 때문에 FCN(Fully Convolutional Network)의 경우 coarse output을 dense pixel로 바꿔줘야 함.**

### FCN 단점 보완: Deconvolution(convolution transpose)
![image](https://user-images.githubusercontent.com/102525066/194113659-4b3d1e59-8944-4457-8909-83ee858f4ded.png)
* Stride를 2로 주면 반으로 줄어듦. zero padding을 안주고!(30×30 -> 15×15로) 이거를 다시 2배로 늘려주는 역할로 생각하기
* **즉, spacial demension을 키워주는 역할을 함**
* 엄밀히 말하면 복원이 될 수는 없음(픽셀 1개마다의 정보를 다시 복원할 수는 없음)
* 다만, 계산 상으로 편하기 때문에 쓰이는 방법정도로 생각하기

## 🧨 Detaction 1: R-CNN
* 목표: FCN 방법 말고 검출하는 방식으로 접근해보자
* 방법
1. 이미지에서 약 2000개의 영역을 뽑아냄(알고리즘에 의해 뽑아냄)
2. 똑같은 크기로 맞춤(AlexNet 이용- Alexnet 2000번 들어간다고 생각하기)
3. Linear SVM을 이용해서 분류
* 단점: 이미지 안에서 2000개의 영역을 뽑으면 2000번을 CNN에 통과시켜야함 -> CPU 처리 속도가 1분이 나옴 -> 해결 필요

## 🧨 Detaction 2: SPPNet(Spatial Pyramid Pooling Network)
* 목표: R-CNN의 단점을 보완해보자 -> CNN을 1번만 돌리게 하자
* 방법
1. 이미지에서 Bounding box를 뽑음
2. 이미지 전체에 대해서 convolution feature map을 만듦
3. 뽑힌 bounding box에 해당하는 convolution feature tensor만 끌고 와서 CNN을 계산(CNN 1번만 사용)

## 🧨 Detaction 3: Fast R-CNN
* 목표: SPPNet과 방식은 비슷하지만 좀 더 좋게 만들어보자 -> Neural Network 써보자
* 방법
1. 이미지에서 bounding box 약 2000개를 추출
2. CNN을 1번 통과 시킴
3. 각각의 region에 대해서 ROI Pooling을 통해 fixed length feature을 뽑음
4. Neural Network를 통해서 bounding box를 어떻게 움직이면 좋을지 라벨을 찾음

## 🧨 Detaction 4: Faster R-CNN
* 목표: bouding box를 뽑아내는 것을 Region Propasal Network를 통해서 학습을 시키자
* 방법: Region Propasal Network + Fast R-CNN
* RPN(Region Propasal Network)의 궁극적인 목표
→ bounding box로써의 의미가 있을지 없을지(물체가 있을지 없을지)를 찾아주는 것
→ 핵심: anchor box의 크기(template 활용도)
![image](https://user-images.githubusercontent.com/102525066/194123085-d55da281-dea5-4d53-b56e-ad84a8ce0e0a.png)
![image](https://user-images.githubusercontent.com/102525066/194123356-63eb6ba4-be25-4699-83ae-554e4f5fea24.png)

→ 여기서도 역시나 FCN은 활용됨 (54개의 channel이 도출 됨: 9*4+2)

## 🧨 Detaction 5: YOLO(v1)
* Bounding box를 뽑는 절차가 없어서 Faster R-CNN보다 훨씬빠름
* 방법
1. 이미지가 들어오면 S×S 격자형태로 나눔
2. 이미지 중앙에 해당 grid 안에 들어가면, 해당 물체의 bounding box와 해당 물체가 무엇인지 같이 예측해줌
3. B(=5)개의 bounding box를 예측해줌(bounding box의 x,y,w,h를 찾아주고 쓸모있는지 없는지 알아냄)
4. 3번과 동시에 Class(C) 예측을 진행
5. 3번과 4번의 정보를 취합해서 box위치와 box가 무엇인지를 동시에 보여주는 형식
* **즉, S×S×(B*5+C) size tesnsor**

##### 참고 강의: https://www.boostcourse.org/ai111
