import tensorflow as tf
import os 
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = tf.keras.models.load_model('my_model.h5')
print(model.summary())

# 이미지 불러와서 집어넣기
img = cv2.imread('Image/temp.png', cv2.IMREAD_GRAYSCALE).reshape(1, 28, 28, 1).astype('float32') / 255.0
print(img.shape)

model.predict_step(img)

# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()