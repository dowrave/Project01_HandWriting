# 이 파일은 코랩에서 실행됨
# phd08_to_npy.py로 생성된 데이터 넘파이 세트로 저장 ########################################################################

# connect google drive
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler

directory = './drive/MyDrive/Project/data/'
# 수정 : 파일 1개당 1개의 label만 들어가게 조정함 - 파일 갯수 * label 수 // 2 가 전체 데이터 수
print(len(os.listdir(directory)))  # 2350 * 2여야 되는 거 아님? 
# print(os.listdir(directory)) 

# 220704 : each_size, width, height 수정
label = 2350
each_size = 81 # 220704 : 100 -> 50개로 축소(코랩 램 이슈)
total_size = label * each_size
img_width = 60
img_height = 60

# 총 arr의 사이즈를 생각해 미리 만들어 둠
image_arr = np.empty((total_size, img_width, img_height), dtype = np.float32) # np.float16은 plt.imshow()가 되지 않는다
label_arr = np.empty(total_size, )

# print(image_arr.shape, label_arr.shape)

# 정규화 후 넘파이 파일로 저장
for i in range(len(os.listdir(directory)) // 2):

    img = np.load(f"./drive/MyDrive/Project/data/phd08_data_{i + 1}.npy")
    label = np.load(f"./drive/MyDrive/Project/data/phd08_labels_{i + 1}.npy")

    for j in range(each_size): 
        scaler = MinMaxScaler()
        img[j] = abs(1 - scaler.fit_transform(img[j])) 
        image_arr[each_size * i + j] = img[j] 
        label_arr[each_size * i + j] = label[j] 
    
    if i % 100 == 0:
      print("Processing : ", np.round(i / (len(os.listdir(directory)) // 2) * 100), " % ")
# print(image.shape, label.shape)

# 넘파이 파일 저장
np.save('./drive/MyDrive/Project/image_arr.npy',image_arr)
np.save('./drive/MyDrive/Project/label_arr.npy',label_arr)

########################################################################################################################

# 데이터 불러오기 & 전처리 & 모델 생성 및 학습 & 저장

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2 

# connect google drive
# from google.colab import drive
# drive.mount('/content/drive')

# dict 파일 불러오기 (코랩 전용)
# !pip install pickle5
# import pickle5 as pickle

image_arr = np.load('/content/drive/MyDrive/Project/image_arr.npy')
label_arr = np.load('/content/drive/MyDrive/Project/label_arr.npy')

# 데이터 숫자 -> 글자 dict
with open('/content/drive/MyDrive/Project/kr_label.pkl', 'rb') as f:
	label_dict = pickle.load(f)

## 데이터 전처리 : 랜덤 축소 -> 이진화 -> 정규화 

min_factor = 0
max_factor = 0.5

data_augmentation = tf.keras.Sequential([
  layers.RandomZoom(height_factor = (min_factor, max_factor), width_factor = (min_factor, max_factor), fill_mode = 'constant', fill_value = 1),
  
])

image_arr = data_augmentation(image_arr.reshape(-1, 60, 60, 1)).numpy()
image_arr = (image_arr * 255).astype('uint8').reshape(-1, 60, 60) # 이진화를 위해선 데이터타입이 uint8이어야 함

# 전처리 이전 시각화
# fig, ax = plt.subplots(2,5)
# for i in range(2):
#   for j in range(5):
#     ax[i][j].imshow(image_arr[200*i + 49*j], cmap = 'gray')
#     ax[i][j].axis('off')

# 
for i in range(image_arr.shape[0]):
  _, image_arr[i] = cv2.threshold(image_arr[i], -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 전처리 결과 확인 
# fig, ax = plt.subplots(2,5)
# for i in range(2):
#   for j in range(5):
#     ax[i][j].imshow(image_arr[200*i + 49*j], cmap = 'gray')
#     ax[i][j].axis('off')

# 훈련 & 테스트 데이터 분리
image_arr = image_arr.reshape(-1, image_arr.shape[1], image_arr.shape[2], 1).astype(np.float32) / 255.0 # 앞에서 255를 곱해줬으니 여기서 다시 정규화해준다

train_img, test_img, train_label, test_label = train_test_split(image_arr, label_arr, test_size = 0.1, stratify = label_arr)
sub_img, val_img, sub_label, val_label = train_test_split(train_img, train_label, test_size = 0.1, stratify = train_label)

# 쓰지 않는 데이터 메모리 해제
del image_arr, label_arr

# 변수들 - num_classes 빼고는 딱히 안쓰게 되었다.
num_classes = 2350
kernel_init = tf.keras.initializers.glorot_uniform()
# bias_init = tf.keras.initializers.Constant(value = 0.2)
bias_init = 'zeros'

# Inception 모듈
def revised_inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                     filters_3x3_reduce_2, filters_3x3_2, filters_3x3_factor, filters_pool_proj, name = None,
                     kernel_init = 'glorot_uniform', bias_init = 'zeros'):
  
  conv_1x1 = layers.Conv2D(filters_1x1, (1,1), padding = 'same', activation = 'relu',
                           kernel_initializer = kernel_init, bias_initializer = bias_init)(x)
                  
  conv_3x3_reduce = layers.Conv2D(filters_3x3_reduce, (1,1), padding = 'same', activation = 'relu',
                           kernel_initializer = kernel_init, bias_initializer = bias_init)(x)
                  
  conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding = 'same', activation = 'relu',
                           kernel_initializer = kernel_init, bias_initializer = bias_init)(conv_3x3_reduce)

  # Factorization_Inception - 5*5 대비 연산량 30% 감소
  conv_3x3_reduce_2 = layers.Conv2D(filters_3x3_reduce_2, (1,1), padding = 'same', activation = 'relu',
                           kernel_initializer = kernel_init, bias_initializer = bias_init)(x)
                  
  conv_3x3_2 = layers.Conv2D(filters_3x3_2, (3, 3), padding = 'same', activation = 'relu',
                           kernel_initializer = kernel_init, bias_initializer = bias_init)(conv_3x3_reduce_2)

  conv_3x3_factor = layers.Conv2D(filters_3x3_factor , (3,3), padding = 'same', activation = 'relu',
                                kernel_initializer = kernel_init, bias_initializer = bias_init)(conv_3x3_2)


  max_pool = layers.MaxPool2D((3,3), strides = (1, 1), padding = 'same')(x)

  pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding = 'same', activation = 'relu', kernel_initializer = kernel_init,
                            bias_initializer = bias_init)(max_pool)

  output = layers.concatenate([conv_1x1, conv_3x3, conv_3x3_factor, pool_proj], axis = 3, name = name)

  return output   


# 전체 모델 
img_size = (60, 60, 1)

input_layer = layers.Input(shape = img_size)

# 층 1: Stem - Feature Map 생성 영역
x = layers.Conv2D(64, (5, 5), strides = (1, 1), name = 'conv_1_5x5/1', activation = 'relu', kernel_initializer = kernel_init,
                  bias_initializer = bias_init)(input_layer)
x = layers.MaxPool2D( (3,3) , strides = (2, 2), name = 'maxpool_1_3x3/2', padding = 'same')(x)
x = layers.BatchNormalization()(x)

# 층 2 : Inception Module a, b
x = revised_inception_module(x, 32, 24, 32, 32, 48, 48, 16, name = "Inception_a")
x = revised_inception_module(x, 64, 48, 64, 64, 96, 96, 32, name = "Inception_b_1")

# MaxPooling
x = layers.MaxPool2D( (3,3) , strides = (2, 2), name = 'maxpool_2_3x3/2', padding = 'same')(x)

# 층 3 : Inception Module b, c
x = revised_inception_module(x, 64, 48, 64, 64, 96, 96, 32, name = "Inception_b_2")
x = revised_inception_module(x, 128, 96, 128, 128, 192, 192, 64, name = 'Inception_c_1')

x = layers.MaxPool2D( (3,3) , strides = (2, 2), name = 'maxpool_2_3x3/3', padding = 'same')(x)

# 층 4 : Inception Module c
x = revised_inception_module(x, 128, 96, 128, 128, 192, 192, 64, name = 'Inception_c_2')

# 층 5 : 분류층까지
x = layers.AveragePooling2D((7,7), strides = (1, 1), padding = 'valid', name = 'avgpool_1_7x7')(x)
x = layers.Dropout(0.4)(x)
x = layers.Flatten()(x)
x = layers.Dense(num_classes)(x)

model = tf.keras.Model(input_layer, x, name = 'Model')

# model.summary() # 학습 파라미터 330만개

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001) # 디폴트

model.compile(optimizer = optimizer,
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['Accuracy'])

lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-4)
es = EarlyStopping(monitor = 'val_loss', patience = 6, restore_best_weights = True)


history = model.fit(sub_img, sub_label, epochs = 100,
          batch_size = 64, # 데이터의 수가 논문 대비 절반이기 때문에 batch_size도 반으로 줄임
          validation_data = (val_img, val_label),
          callbacks = [lr, es])

# val_loss: 0.0038 - val_Accuracy: 0.9992 - lr: 5.0000e-04 에서 멈춤

model.evaluate(test_img, test_label) # loss: 0.0010 - Accuracy: 0.9995

model.save('korean_model_220707.h5')
