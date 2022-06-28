import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

plt.figure()
plt.imshow(train_images[0], cmap='gray')
plt.xlabel([train_labels[0]])
plt.grid(False)
plt.show()


# 데이터 전처리 : 정규화, 흑백반전, 채널 차원 추가

train_images = (255 - train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')) / 255.0
sub_images, val_images, sub_labels, val_labels = train_test_split(train_images, train_labels, test_size = 0.2)

test_images = (255 - test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')) / 255.0

print(train_images.shape)

# 모델
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

model.fit(train_images, train_labels, epochs = 20,
          validation_data = (val_images, val_labels),
          callbacks=  [es])

# 테스트 데이터에 대한 추론
model.evaluate(test_images, test_labels)

# 모델 저장
model.save('model_220624.h5')