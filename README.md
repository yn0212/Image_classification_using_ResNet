# Image classification using ResNet :pencil2: :computer:
![Footer](https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=footer)

# :pushpin:Project Description
ResNet50 신경망을 사용하고 완전연결층을 직접 추가해 로컬데이터의 필기체를 95%확률로 인식하는 프로그램이다.

# :pushpin:Project Purpose
- 이미지 분류를 위해 신경망 ResNet50를 사용해 필기체 인식률을 95%이상 높이기
- 전이학습 이용해 이미지넷의 데이터셋을 써서 훈련된 모델의 가중치를 가져와 필기체 인식 과제에 맞게 보정해서 사용하기위함
- 사전 훈련된 네트워크의 합성곱층(가중치 고정)에 새로운 데이터를 통과시키고 , 완전연결층을 조절해 그 출력을 데이터 분류기에서 훈련시켜보며 공부한 내용 복습

# :pushpin:Device used for the project
- 학습에 사용한 GPU : GTX1660 Supper

# :pushpin:Project Dataset
- 사용한 데이터셋 : 직접 만든 로컬 데이터
- 클래스 수 : 0~9 총 10개(훈련 데이터 : 약 400-500개 , 검증 데이터: 약 60개)

# :pushpin:Project Results
- 필기체 인식률 : 94.73%

![47af2254e7b22686fa36668e35342adcd7aaa0bd_re_1673242484456](https://user-images.githubusercontent.com/105347300/211261827-2b724914-3208-4d45-93fd-0978c7fd3d0b.png)

- 100개의 검증이미지 중 97개 맞춤 
- epochs =  600

![3ca97d112b2d77791febca5fa9d99e187ac99e86_re_1673242484456](https://user-images.githubusercontent.com/105347300/211261907-ca09c351-99f0-4cd6-be6a-206c553883ae.png)

- 훈련데이터 정확도: 0.9411
- 훈련데이터 오차:0.0323
- 검증데이터 정확도:0.9473
- 검증데이터 오차:0.0262

# :pushpin:Project Process
## :loudspeaker: ResNet 정의

- 마이크로소프트에서 개발한 알고리즘으로 “Deep Residual Learning for Image Recognition”이라는 논문에서 발표됨
- 핵심:  깊어진 신경망을 효과적으로 학습하기 위한 방법으로 레지듀얼(residual) 개념을 고안

## :loudspeaker: ResNet 구조

![62ce2e7a7a5f512af0d7d85d6483d6ba5c8186df_re_1673242484454](https://user-images.githubusercontent.com/105347300/211262100-a23407c4-ee4c-42e3-97e2-34ba034d1938.png)

- ==>이 사전 훈련된 합성곱층의 가중치를 고정해서 사용할 것임

- ==>완전연결층의 구조를 조작해보며 그에따른 성능을 비교해 봄

## :loudspeaker: 과정
### :bulb: -ResNet50 모델을 사용해 가중치 고정하고, 밀집층 추가하기

- ==>이진분류 모델에서는 sigmoid 함수를 활성화 함수로 사용하고,다중 분류 모델에서는 softmax 함수를 활성화 함수로 사용
- ==>activation='softmax'로 설정함
- ==>activation='relu' 로 설정시 경사하강법에 영향을 주지않아 속도가 빠르고 기울기 소멸문제가 발생하지 않음.

### :bulb: 모델 훈련 환경설정 , 데이터 전처리
- ImageDataGenerator - 객체 생성==> 내 데이터셋의 양이 적으므로 전처리 과정을 통해 이미지를 증식함.

- flow_from_directory-객체 생성 ==> 폴더 구조를 그대로 가져와서 imageDataGenerator에 실제 데이터를 채워줌.(자신의 로컬 폴더에있는 데이터를 사용하기 위해 생성함.)

### :bulb: 모델 훈련 

- 시행착오 결과 epoch는 600으로 결정.

### :bulb: 예측
- 검증데이터 100개를 랜덤으로 가져와 예측이 틀리면 빨간색, 맞으면 노란색 글씨로 출력
- 시행착오 결과 epoch를 충분히 적절히 설정해야함
- 최적의 epochs =100으로 설정함.

### :bulb: 예측 후 값 설정 과정

#### 1. 완전연결층의 수 , epoch 값 비교
- 완전 연결층이 하나일때 : 모델 성능 좋아지지 않음 

![bc1a384e08fa0bc6cc4b7334dfe59f8a5eca7094_re_1673242484455](https://user-images.githubusercontent.com/105347300/211262585-712505f1-06c3-436e-aeef-7eb3f105e623.png)


- 완전연결층을 늘림 ==>성능이 향상됨.

![75bbfdfc7d2ea9ae9fef83954e6162422bfe2d7b_re_1673242484455](https://user-images.githubusercontent.com/105347300/211262621-9bddcaa6-b461-4355-9c8b-7a665c874aff.png)

:bulb: ==> 시행착오 결과 밀집층이 많고 epoch를 충분히 적절히 늘릴수록 성능이 좋아짐.

#### 2.동일조건에서 완전연결층에 드롭 아웃해 과적합을 줄인 결과
- 드롭아웃 안 한 모델 : epoch =150일때
![40f0470d9676761c5c489d248c84cf2e4d318ce8_re_1673242484455](https://user-images.githubusercontent.com/105347300/211262776-0ed4274f-a6df-4533-9c8c-9c664f7df194.png)

![af51501f0d3dc1dbe0d96507a2d7c790f1b3bc86_re_1673242484455](https://user-images.githubusercontent.com/105347300/211262785-8a3732f3-563e-4232-8714-f59407b4ff35.png)

- 완전연결층에 드롭아웃 적용한 모델 , epoch =150

![1aebc71595db434d99621df5f0294b27f4a06c45_re_1673242484455](https://user-images.githubusercontent.com/105347300/211262818-54441f55-d791-47c5-ab1c-145238a9c8aa.png)

![8ecc5f28a15bb889fc02aea1faa4ec448c52d6c5_re_1673242484456](https://user-images.githubusercontent.com/105347300/211262848-5a1313ba-f08e-4390-bd1a-68db957b8d53.png)

- 드롭아웃한 모델은 드롭아웃 안 한 모델보다  검증데이터셋의 예측률은 더 좋게나옴

:bulb: ==>과적합 방지 됨!

#### 드롭아웃 한 모델 : epoch=150일때

- 훈련 데이터셋  오차 :0.0741      정확도 : 0.8603

- 검증 데이터셋  오차: 0.0709      정확도:  0.8740

#### 드옵아웃 안 한 모델 : epoch =150일때

- 훈련 데이터셋  오차 :0.0606      정확도 : 0.8851

- 검증 데이터셋  오차: 0.0928      정확도:  0.8293



:bulb: ==>검증데이터의 오차가 줄어든 드롭아웃한 모델을 선택하기로 결정 후 정확도를 높이기위해 훈련데이터를 추가함.

### :bulb: 최종 결과
-모델 파일 이름:

![47af2254e7b22686fa36668e35342adcd7aaa0bd_re_1673242484456](https://user-images.githubusercontent.com/105347300/211263184-95fd1564-e931-4047-8864-3f435b17272d.png)
- epochs =  600
- 필기체 인식률 : 94.73%
- 100개의 검증이미지 중 97개 맞춤 

![3ca97d112b2d77791febca5fa9d99e187ac99e86_re_1673242484456](https://user-images.githubusercontent.com/105347300/211263240-8c8eb241-866f-411b-9ebf-7bfa9b9d90a8.png)

- 훈련데이터 정확도: 0.9411

- 훈련데이터 오차:0.0323

- 검증데이터 정확도:0.9473

- 검증데이터 오차:0.0262

