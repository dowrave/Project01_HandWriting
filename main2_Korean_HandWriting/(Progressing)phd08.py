# 코랩에서 실행됨. 데이터 전처리부터 모델 생성까지

# connect google drive
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


directory = './drive/MyDrive/Project/data/'
# 수정 : 파일 1개당 1개의 label만 들어가게 조정함 - 파일 갯수 * label 수 // 2 가 전체 데이터 수
print(len(os.listdir(directory)))  # 2350 * 2여야 되는 거 아님? 
# print(os.listdir(directory)) 

# 220704 : each_size, width, height 수정
label = 2350
each_size = 50 # 220704 : 100 -> 50개로 축소(코랩 램 이슈)
total_size = label * each_size
img_width = 60
img_height = 60

# 총 arr의 사이즈를 생각해 미리 만들어 둠
image_arr = np.empty((total_size, img_width, img_height), dtype = np.float32) # np.float16은 plt.imshow()가 되지 않는다
label_arr = np.empty(total_size, )

print(image_arr.shape, label_arr.shape)

# 그냥 np.append를 이용하면 2차원 배열이 1차원으로 들어감
for i in range(len(os.listdir(directory)) // 2):

    img = np.load(f"./drive/MyDrive/Project/data/phd08_data_{i + 1}.npy")
    label = np.load(f"./drive/MyDrive/Project/data/phd08_labels_{i + 1}.npy")

    for j in range(each_size): 
        scaler = MinMaxScaler()
        img[j] = abs(1 - scaler.fit_transform(img[j])) # 스케일링 & 흑백 반전
        image_arr[each_size * i + j] = img[j] # each_size * i 면 뒷쪽의 절반이 누락될 수 밖에 없음
        label_arr[each_size * i + j] = label[j] 

    # Concatenate : Arr을 계속 리사이즈하기 때문에 느림.
    
    if i % 100 == 0:
      print("Processing : ", np.round(i / (len(os.listdir(directory)) // 2) * 100), " % ")

# 색반전 & 정규화 & 병합 넘파이 파일 저장
np.save('./drive/MyDrive/Project/image_arr.npy',image_arr)
np.save('./drive/MyDrive/Project/label_arr.npy',label_arr)

# 모델 불러오기 ~ 모델 생성 및 훈련

image_arr = np.load('/content/drive/MyDrive/Project/image_arr.npy')
label_arr = np.load('/content/drive/MyDrive/Project/label_arr.npy')

image_arr = image_arr.reshape(-1, image_arr.shape[1], image_arr.shape[2], 1)

train_img, test_img, train_label, test_label = train_test_split(image_arr, label_arr, test_size = 0.1, stratify = label_arr)
sub_img, val_img, sub_label, val_label = train_test_split(train_img, train_label, test_size = 0.1, stratify = train_label)

# 쓰지 않는 데이터들의 메모리 할당을 없애줌 -> 데이터 사이즈 자체가 줄면서 안 쓸지도?
del image_arr, label_arr

# 모델 생성

# 변수들
num_classes = 2350
kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value = 0.2)

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

# 전체 모델 구성

img_size = (60, 60, 1)

input_layer = layers.Input(shape = img_size)

# 층 1 - Stem
x = layers.Conv2D(64, (5, 5), strides = (1, 1), name = 'conv_1_5x5/1', kernel_initializer = kernel_init,
                  bias_initializer = bias_init)(input_layer)
x = layers.BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = layers.MaxPool2D( (3,3) , strides = (2, 2), name = 'maxpool_1_3x3/2', padding = 'same')(x)

# 층 2 : Inception Module
x = revised_inception_module(x, 32, 24, 32, 32, 48, 48, 16, name = "Inception_a")
x = revised_inception_module(x, 64, 48, 64, 64, 96, 96, 32, name = "Inception_b_1")

# MaxPooling
x = layers.MaxPool2D( (3,3) , strides = (2, 2), name = 'maxpool_2_3x3/2', padding = 'same')(x)

# 층 3 : Inception Module
x = revised_inception_module(x, 64, 48, 64, 64, 96, 96, 32, name = "Inception_b_2")
x = revised_inception_module(x, 64, 48, 64, 64, 96, 96, 32, name = "Inception_b_3")
x = revised_inception_module(x, 64, 48, 64, 64, 96, 96, 32, name = "Inception_b_4")
x = revised_inception_module(x, 64, 48, 64, 64, 96, 96, 32, name = "Inception_b_5")
x = revised_inception_module(x, 128, 96, 128, 128, 192, 192, 64, name = 'Inception_c_1')

x = layers.MaxPool2D( (3,3) , strides = (2, 2), name = 'maxpool_2_3x3/3', padding = 'same')(x)

# 층 4
x = revised_inception_module(x, 128, 96, 128, 128, 192, 192, 64, name = 'Inception_c_2')
x = revised_inception_module(x, 128, 96, 128, 128, 192, 192, 64, name = 'Inception_c_3')

x = layers.GlobalAveragePooling2D(name = 'GlobalAvgPooling')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(num_classes)(x)

model = tf.keras.Model(input_layer, x, name = 'Model')

# model.summary()

# es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-4)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

model.compile(optimizer = optimizer,
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['Accuracy'])

model.fit(sub_img, sub_label, epochs = 100,
          batch_size = 128,
          validation_data = (val_img, val_label),
          callbacks = [lr])