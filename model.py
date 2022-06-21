import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 데이터 전처리 : 채널 하나 더 만들고, 0~1 로 정규화
train_images = train_images / 255.0
train_images = train_images.reshape(train_images.shape[0], 28, 28 ,1).astype('float32')

test_images = test_images / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28 ,1).astype('float32')

# 데이터 배치 만들고 섞기
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# 모델 정의

class MyCNNModel(tf.keras.Model):
  def __init__(self):
    super(MyCNNModel, self).__init__()
    self.conv1 = layers.Conv2D(32, 3, padding = 'same')
    self.conv2 = layers.Conv2D(64, 3, padding = 'same')
    self.conv3 = layers.Conv2D(128, 3, padding = 'same')
    self.flatten = layers.Flatten()
    # self.batchnorm = layers.BatchNormalization()
    self.lrelu = layers.LeakyReLU()
    self.dropout = layers.Dropout(0.3)
    self.d1 = layers.Dense(128, activation = 'relu')
    self.d2 = layers.Dense(10)

  def call(self, x):
    x = self.conv1(x) # 함수형으로 만들면 input_shape 설정 필요 없나??
    # x = self.batchnorm(x)
    x = self.lrelu(x)
    x = self.dropout(x)
    x = self.conv2(x)
    # x = self.batchnorm(x)
    x = self.lrelu(x)
    x = self.dropout(x)
    x = self.conv3(x)
    # x = self.batchnorm(x)
    x = self.lrelu(x)
    x = self.dropout(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = MyCNNModel()

# 옵티마이저 & 손실 함수 정의
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam()

# 모델 손실, 성능 측정 지표
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_acc')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_acc')

# 모델 훈련
@tf.function 
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images, training = True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_acc(labels, predictions)
  return loss

# 추론
@tf.function
def test_step(images, labels):
  predictions = model(images, 
                      training = False) # Dropout 같은 레이어는 자동으로 꺼짐
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_acc(labels, predictions)

# 전체 훈련 과정

def train(EPOCHS = 10):

  for epoch in range(1, EPOCHS + 1):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()

    for images, labels in train_dataset:
      train_step(images, labels)

    # 에포크에 대한 체크포인트를 작성하려면 여기에서 해야 함 - 체크포인트 저장
    if epoch % 5 == 0 :
      model.save_weights(f'save_epoch_{epoch}')
      print(f'에포크 {epoch}의 모델 저장됨.')

    for test_images, test_labels in test_dataset:
      test_step(test_images, test_labels)

    print(f'Epoch {epoch}', 
          f'Loss : {train_loss.result()}',
          f"Accuracy : {train_acc.result() * 100}%",
          f'Test Loss : {test_loss.result()}',
          f'Test Accuracy : {test_acc.result() * 100}%'
          )

train(20)

model.save('my_model.tf') 
model.summary()

# 모델 로드:
# new_model = tf.keras.models.load_model('my_model.tf')
# new_model.summary()