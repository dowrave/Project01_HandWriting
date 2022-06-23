# 구동은 코랩에서 했음. 
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

plt.figure()
plt.imshow(train_images[0])
plt.xlabel([train_labels[0]])
plt.grid(False)
plt.show()

# 데이터 전처리 : 채널 추가, 정규화
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255.0
sub_images, val_images, sub_labels, val_labels = train_test_split(train_images, train_labels, test_size = 0.2)

test_images = test_images.reshape(test_images.shape[0], 28, 28 , 1).astype('float32') / 255.0

# 데이터 배치 만들고 섞기(For Low Level)
# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
# test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# 그냥 케라스로 모델 간단하게 정의하겠음. 상속으로 만든 모델은 조금 다르게 손이 가는 부분이 많음
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding = 'same', input_shape = (28, 28, 1),
                           activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10)
])

es = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
# reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-4)

model.compile(optimizer = 'adam', # adam 디폴트 lr : 1e-3
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), # 결과 : 정수, softmax 사용 안해도 됨
              metrics = ['accuracy'],)

# 테스트 데이터에 대한 추론
model.evaluate(test_images, test_labels)

# 모델 저장
model.save('my_model.h5')

# 모델 불러오기
loaded_model = tf.keras.models.load_model('my_model.h5')
loaded_model.summary()