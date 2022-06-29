# 모델 실행은 구글 코랩에서 진행, 모델만 다운받음

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
# import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 데이터 전처리 : 0~1 로 정규화
# plt.figure()
# plt.imshow(train_images[0], cmap='gray')
# plt.xlabel([train_labels[0]])
# plt.grid(False)
# plt.show()

# 전처리
train_images = (255 - train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')) / 255.0
sub_images, val_images, sub_labels, val_labels = train_test_split(train_images, train_labels, test_size = 0.2)

test_images = (255 - test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')) / 255.0

# 이미지 증강
img_augmentation = tf.keras.Sequential([
                                        tf.keras.layers.RandomZoom(height_factor = (-0.2, +0.2),
                                                                   width_factor = (-0.2, +0.2)),
                                        tf.keras.layers.RandomRotation(0.3)
])

# 모델 생성 & 컴파일 & 실행
# 그냥 케라스로 모델 간단하게 정의
model = tf.keras.Sequential([
    # 220628 : 증강층 추가
    img_augmentation,

    tf.keras.layers.Conv2D(32, (3, 3), padding = 'same',
                           activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10)
])

es = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)

model.compile(optimizer = 'adam', # adam 디폴트 lr : 1e-3 인데 LROnplateau를 안 쓰게 되어서 상관없어짐
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), # 결과 : 정수, softmax 사용 안해도 됨
              metrics = ['accuracy'],)

model.fit(train_images, train_labels, epochs = 30,
          validation_data = (val_images, val_labels),
          callbacks=  [es])

# loss: 0.1446 - accuracy: 0.9565 - val_loss: 0.0504 - val_accuracy: 0.9854 (16에포크에서 최고 값)
# Underfitting이긴 한데, 이미지 증강 층 때문에 발생한 현상으로 보임

# 테스트 데이터로 성능 평가
model.evaluate(test_images, test_labels) # loss: 0.0491 - accuracy: 0.9847

# 모델 저장
model.save('model_220628.h5')

# 모델 불러오기
# loaded_model = tf.keras.models.load_model('220624.h5')
# loaded_model.summary()