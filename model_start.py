import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
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
 
 
font_fname = 'C:/Windows/Fonts/malgun.ttf'
font_family = font_manager.FontProperties(fname=font_fname).get_name()
 
history = keras.models.load_model('model_150_2.h5')
 
 
BATCH_SIZE = 32
image_height = 224
image_width = 224
train_dir = './data/num/train'  #경로 설정
valid_dir = './data/num/test1'
 
 
 
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
 
 
 
 

class_names = ['0','1','2','3','4','5','6','7','8','9']
validation, label_batch = next(iter(valid_generator)) #validation: 이미지 배열, label_batch : 정답 레이블 배열
#반복자(iterator)를 사용
#iter() 메서드는 전달된 데이터의 반복자를 꺼내 반환
#next() 메서드는 반복자를 입력으로 받아 그 반복자가 다음에 출력해야 할 요소를 반환
# iter() 메서드로 반복자를 구하고 그 반복자를 next() 메서드에 전달하여 차례대로 꺼내기
 
prediction_values = history.predict(validation) #예측값
prediction_values = np.argmax(prediction_values, axis=1) #이미지 배열의 예측값
 
fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
for i in range(32):
    ax = fig.add_subplot(6, 6, i+1, xticks=[], yticks=[])
    ax.imshow(validation[i,:], cmap=plt.cm.gray_r, interpolation='nearest')
    print('prediction_values[i]=',prediction_values[i])
    print('np.argmax(label_batch[i])=',np.argmax(label_batch[i]))
    if prediction_values[i] == np.argmax(label_batch[i]): #이미지배열 예측값==정답 레이블배열의 최대값인덱스
        ax.text(3, 17, class_names[prediction_values[i]], color='yellow', fontsize=14)
    else:
        ax.text(3, 17, class_names[prediction_values[i]], color='red', fontsize=14)
plt.show()
