import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
 
from tensorflow.python.keras import Model
from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GlobalMaxPool2D, GlobalAveragePooling2D ,Dropout
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
#정확도 시각화
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib import font_manager
 
#예제에서 사용할 모델은 사전 훈련된 ResNet50을 사용
# include_top:네트워크 상단에 완전연결층을 포함할지 여부를 지정``
model = ResNet50(include_top=True,
#weights: 가중치,None:무작위 초기화,imagenet:ImageNet에서 사전 훈련된 값
                 weights="imagenet",
                 input_tensor=None, #input_tensor: 입력 데이터의 텐서
                 input_shape=None,#input_shape: 입력 이미지에 대한 텐서 크기
                 #풀링,None: 마지막 합성곱층 출력,
                 #avg: 마지막 합성곱층에 글로벌 평균 풀링이 추가
                 #max: 마지막 합성곱층에 글로벌 최대 풀링이 추가
                 pooling=None,
#weights로 'imagenet'을 사용시 classes 값이 1000이어야함. 다른 값으로 사용시 'None'지정
                 classes=1000)
 
 
#사전 훈련된 합성곱층의 가중치를 고정
model.trainable = False 
#사전 훈련된 합성곱층의 가중치를 고정, 
# 시그모이드 활성화 함수가 포함된 밀집층(완전연결층)을 추가된 모델
 
model = Sequential([model,Dense(500, activation='relu'),Dense(250, activation='relu'),(Dropout(0.2)),
                    Dense(125, activation='relu'), Dense(62, activation='relu'),Dropout(0.2)
                    ,Dense(31, activation='relu'),Dense(10, activation='softmax')])  # 밀집층 추가
 
#모델 훈련 환경설정
model.compile(loss='binary_crossentropy',
##loss : 최적화 과정에서 사용될 손실함수 설정. 
#optimizer : 손실함수를 사용해 구한 값으로 기울기를 구하고
#  신경망의 파라미터를 학습에 어떻게 반영할지 결정하는 방법
              optimizer='adam', #adam손실함수 사용
              metrics=['accuracy'])#metrics:모델의 평가기준 지정
 
BATCH_SIZE = 100
image_height = 224
image_width = 224
train_dir = './data/num/train'  #경로 설정
valid_dir = './data/num/test1'
 
# ImageDataGenerator 사용시 파라미터 설정을 사용해 데이터 전처리 쉽게 할 수 있음.
train = ImageDataGenerator(
                 rescale=1./255, #1/255로 스케일링하여 0~1 범위로 변환
                 rotation_range=10, #이미지 회전 범위 10은 0~10도 범위 내 임의로 회전
#그림을 수평으로 랜덤하게 평행 이동시키는 범위
                 width_shift_range=0.1,
 #0.1은 전체 넓이가 100일 경우, 0.1의 값을 적용해 10픽셀 내외로 좌우로 이동
 #그림을 수직으로 랜덤하게 평행 이동시키는 범위
                 height_shift_range=0.1
#0.1은 전체 넓이가 100일 경우, 0.1의 값을 적용해 10픽셀 내외로 상하로 이동
 
#0.1라디안 내외로 시계 반대 방향으로 이미지를 변환
) 
#임의 확대/축소 범위. 0.1은 0.9에서 1.1배의 크기로 이미지를 변환
 
#flow_from_directory 메서드:
#   폴더 구조를 그대로 가져와서 ImageDataGenerator에 실제 데이터를 채워 줌
train_generator = train.flow_from_directory(train_dir, #훈련 이미지 경로
                                    
                                        ##이미지크기,모든 이미지는 이 크기로 자동 조정
                                            target_size=(image_height, image_width), 
                                        #그레이스케일이면 'grayscale', 색상 'rgb' 사용
                                            color_mode="rgb",
                                        #배치당 generator에서 생성할 이미지 개수
                                            batch_size=BATCH_SIZE,
                                        #이미지를 임의로 섞기 위한 랜덤한 숫자
                                            seed=1,
                                        #이미지 섞어서 사용하려면 shuffle을 True
                                            shuffle=True,
#예측할 클래스가 두 개뿐이라면 "binary"를 선택,아니면 "categorical"을 선택
                                            class_mode="categorical") 
 
# ImageDataGenerator 사용시 파라미터 설정을 사용해 데이터 전처리 쉽게 할 수 있음.
valid = ImageDataGenerator(rescale=1.0/255.0)
#flow_from_directory 메서드:  폴더 구조를 그대로 가져와서
# ImageDataGenerator에 실제 데이터를 채워 줌
valid_generator = valid.flow_from_directory(valid_dir, #테스트 이미지 경로
                                            
                                            target_size=(image_height, image_width),
                                            color_mode="rgb",
                                            batch_size=BATCH_SIZE,
                                            seed=7,
                                            shuffle=True,
                                            class_mode="categorical")
                                            
#모델 훈련
history = model.fit(train_generator, #학습에 사용되는 데이터셋
                    epochs=600,#학습에 대한 반복 횟수
                    validation_data=valid_generator, #검증 데이터셋을 설정
                    #련의 진행 과정을 보여 줌
    #0이면 아무것도 출력x, 1: 진행도 진행 막대 ,2: 미니 배치마다 훈련 정보
                    verbose=2) 
 
font_fname = 'C:/Windows/Fonts/malgun.ttf'
font_family = font_manager.FontProperties(fname=font_fname).get_name()
 
 
model.save('model_150_2.h5')
 
plt.rcParams["font.family"] = font_family
#accuracy: 매 에포크에 대한 훈련의 정확도
accuracy = history.history['accuracy'] 
#val_accuracy: 매 에포크에 대한 검증의 정확도
val_accuracy = history.history['val_accuracy']
#loss: 매 에포크에 대한 훈련의 손실 값
loss = history.history['loss']
# val_loss: 매 에포크에 대한 검증의 손실 값
val_loss = history.history['val_loss']
 
epochs = range(len(accuracy))
 
plt.plot(epochs, accuracy, label="훈련 데이터셋")
plt.plot(epochs, val_accuracy, label="검증 데이터셋")
plt.legend()
plt.title('정확도')
plt.figure()
 
plt.plot(epochs, loss, label="훈련 데이터셋")
plt.plot(epochs, val_loss, label="검증 데이터셋")
plt.legend()
plt.title('오차')
 

class_names = ['0','1','2','3','4','5','6','7','8','9']
validation, label_batch = next(iter(valid_generator)) #validation: 이미지 배열, label_batch : 정답 레이블 배열
#반복자(iterator)를 사용
#iter() 메서드는 전달된 데이터의 반복자를 꺼내 반환
#next() 메서드는 반복자를 입력으로 받아 그 반복자가 다음에 출력해야 할 요소를 반환
# iter() 메서드로 반복자를 구하고 그 반복자를 next() 메서드에 전달하여 차례대로 꺼내기
 
prediction_values = model.predict(validation) #예측값
prediction_values = np.argmax(prediction_values, axis=1) #이미지 배열의 예측값
 
fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
for i in range(100):
    ax = fig.add_subplot(10, 10, i+1, xticks=[], yticks=[])
    ax.imshow(validation[i,:], cmap=plt.cm.gray_r, interpolation='nearest')
    print('prediction_values[i]=',prediction_values[i])
    print('np.argmax(label_batch[i])=',np.argmax(label_batch[i]))
    if prediction_values[i] == np.argmax(label_batch[i]): #이미지배열 예측값==정답 레이블배열의 최대값인덱스
        ax.text(3, 17, class_names[prediction_values[i]], color='yellow', fontsize=14)
    else:
        ax.text(3, 17, class_names[prediction_values[i]], color='red', fontsize=14)
plt.show()