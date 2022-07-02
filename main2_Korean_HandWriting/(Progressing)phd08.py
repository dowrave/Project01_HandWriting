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

directory = './drive/MyDrive/Project/data/'
print(len(os.listdir(directory)))

label = 2350
each_size = 2187
total_size = label * each_size
img_size = 15

# 총 arr의 사이즈를 생각해 미리 만들어 둠
image_arr = np.empty((total_size, 15, 15), dtype = np.float16)
label_arr = np.empty(total_size, )

print(image_arr.shape, label_arr.shape) # (5139450, 15, 15) (5139450,)

# 그냥 np.append를 이용하면 2차원 배열이 1차원으로 들어감

for i in range(len(os.listdir(directory)) // 2):

    img = np.load(f"./drive/MyDrive/Project/data/phd08_data_{i + 1}.npy")
    label = np.load(f"./drive/MyDrive/Project/data/phd08_labels_{i + 1}.npy")
    for j in range(each_size * 2):
        scaler = MinMaxScaler()
        scaler.fit_transform(img[j])
        image_arr[each_size * 2 * i + j] = img[j] # each_size * i 면 뒷쪽의 절반이 누락될 수 밖에 없음
        label_arr[each_size * 2 * i + j] = label[j] 

    # image_arr = np.concatenate([image_arr, image])
    # label_arr = np.concatenate([label_arr, label])
    # print(image_arr.shape, label_arr.shape) # 4만개 넘어가면 확 느려지네
    
    # 결과 데이터셋의 크기를 알고 있다면 이를 초기화해줄 수 있음 : 위에서 np.empty(size)로 생성
    
    if i % 100 == 0:
      print("Processing : ", i)

# 이미지 확인 : 정규화 & 색 반전
# print(image_arr[0])
# plt.imshow(np.array(image_arr[0], dtype = np.float32), cmap = 'gray') # np.float16은 plt.imshow()가 되지 않음..!

# np.save('./image_arr.npy',image_arr)
# np.save('./label_arr.npy',label_arr)

