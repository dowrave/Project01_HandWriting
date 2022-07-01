# 모델을 만드는 파일...이 될 예정이었으나 모델의 정확도가 너무 낮아서 폐기.
# 시행착오를 그대로 남겨서 주석이 많습니다.
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os
import cv2


# Json 파일 & 따로 다운 받은 이미지들의 픽셀값들을 합쳐서 csv파일로 저장  #####################################
with open('./handwriting_data_info_clean.json', 'r', encoding = 'UTF-8') as f:
    data = json.load(f)

    temp = data['annotations']

    syl_dict = {}
    for d in temp:
        if d['attributes']['gender'] and d['attributes']['type'] == '글자(음절)' and int(d['image_id']) <= 192280:
            syl_dict[d['image_id']] = d['text']

print(len(syl_dict.items())) # 135840 

df = pd.DataFrame({"image_id" : syl_dict.keys(), 
                   "label" : syl_dict.values(),
                   })

# img_arr = []
img_arr = np.empty((0, 625))
IMG_SIZE = 25

# 여러 파일 불러오기 -> 크기 변환하기 -> 배열로 저장 - 파일에 있는 데이터 수 : 135841개
# 데이터프레임 생성 시간에 5시간 넘게 걸렸기 때문에 실행하는 걸 권장하지 않음
for i in os.listdir('./1_syllable/'):
    path = './1_syllable/' + i

    if i == '00192258.png': # JSON파일에 없어서 제외
        continue

    if int(i.rstrip('.png')) % 10000 == 0: # 경과 표시
        print(i) # *.png
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    # print(img.shape) # (세로, 가로)

    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA).flatten() #INTER_AREA는 영상 축소 시 효율적, 디폴트는 LINEAR
    
    img_arr = np.append(img_arr, [resized_img], axis = 0)

img_df = pd.DataFrame(img_arr)

new_df = pd.merge(df, img_df, left_index = True, right_index = True)
print(new_df)
new_df.to_csv('220629_size25.csv')

###############################################################################################################

# 저장한 csv파일을 불러와서 main1.py 에서 만든 모델 생성 및 실행까지 진행
img_df = pd.read_csv('/content/drive/MyDrive/Project/220629_size25.csv', index_col = 'image_id')
img_df.drop('Unnamed: 0', axis = 1, inplace = True)
# img_df.shape # 135840, 626

a = Counter(img_df['label'])
# a = dict(a)

def syl_to_count(keys, dic):
  return dic[keys]

img_df['count'] = img_df['label'].apply(syl_to_count, args = (a,))

# count < 5 인 행들 목록 삭제하기
drop_idx = img_df[img_df['count'] < 5].index # 144개
img_df.drop(drop_idx, inplace = True)

# img_df.shape # 135696, 627

# LabelEncoding
a = Counter(img_df['label'])
character_lst = np.array(list(a.keys())) # dict_keys()에서 arr 변환이 바로 안되는 것 같음

encoder = LabelEncoder()
# print(img_df['label'].values)
encoder.fit(character_lst) 
label_no = encoder.transform(img_df['label'])
img_df['NumericLabel'] = label_no

# img_df['label'].apply도 써봤으나 1차원 array를 넣으라는 오류가 자꾸 떠서 그냥 따로 빼서 진행함

# 단일 샘플 확인
# example = img_df.iloc[0][1:].astype('float').to_numpy().reshape(25, 25)
# print(example)

# plt.imshow(example, cmap = 'gray')
# plt.axis('off')

# label의 분포도 확인 - label이 너무 많아서 직관적이지 않음
# sns.countplot(x = 'label', data = img_df)
# plt.show()

# 정규화 & 범위 줄이기 - 판다스 문법 주의!
images = img_df.iloc[:, 1:-2]
labels = img_df.iloc[:, -1]

# print(images.head())
# print(labels.head())

# len(np.unique(labels))

images = images.to_numpy().reshape(-1, 25, 25, 1) / 255.0
labels = labels.to_numpy()

# 각 글자가 7 ~ 13개 있음 : stratify는 필요할 듯? - 일단 KFold 없이 Stratify만 적용해봄
train_images, test_images, train_label, test_label = train_test_split(images, labels, test_size = 0.2, stratify = labels)
sub_images, val_images, sub_label, val_label = train_test_split(train_images, train_label, test_size = 0.2, stratify = train_label)

# print(sub_images.shape, val_images.shape, test_images.shape)



img_augmentation = tf.keras.Sequential([
                                        tf.keras.layers.RandomZoom(height_factor = (-0.1, +0.1),
                                                                   width_factor = (-0.1, +0.1)),
                                        tf.keras.layers.RandomRotation(0.3)
])

# main1.py에서 사용한 모델을 거의 그대로 가져왔음. 마지막 출력층의 노드 수만 전처리 후 남은 label 수를 집어넣음.

model = tf.keras.Sequential([

    img_augmentation,
    tf.keras.layers.Conv2D(32, (3, 3), padding = 'same',
                           activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(11117)
])

es = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, min_lr = 1e-4)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1) # 초기 정확도가 절망적이라 ReduceLROnPlateau 적용 및 lr 올려봄

model.compile(optimizer = optimizer, 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'],)

model.fit(sub_images, sub_label, epochs = 30,
          validation_data = (val_images, val_label),
          callbacks=  [es])

# Loss 값이 9, 정확도 단위가 1e-5 여서 포기. 다른 모델과 데이터를 찾아야 할 것 같다.