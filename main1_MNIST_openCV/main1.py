import numpy as np
import cv2
import tensorflow as tf 
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 드래그를 구현하려면 LBUTTON_DOWN 때 별도의 파라미터를 on/off 식으로 바꿔주면 됨
drawing = False
pt_x, pt_y = None, None 

# 모델 불러오기
model = tf.keras.models.load_model('./main1_MNIST_openCV/model_220628.h5')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# 마우스 그리기
def draw_lines(event, x, y, flags, param):

    global drawing, pt_x, pt_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt_x, pt_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt_x, pt_y), (x, y), color = BLACK, thickness = 2, lineType = cv2.LINE_AA)
            pt_x, pt_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt_x, pt_y), (x, y), color = BLACK, thickness = 2, lineType = cv2.LINE_AA)

# 일단 MNIST 필기체 인식부터 시작. 바탕은 흰색으로 ㄱㄱ
img = np.ones((100, 100, 1), dtype = np.uint8) * 255
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_lines)

while True:
    cv2.imshow('image', img)

    # 구현은 필기체 작성 -> 이미지를 추론 모델에 집어넣음 -> 예측 결과를 반환 하는 방식으로 이루어질 듯

    if cv2.waitKey(5) == ord('q') or cv2.waitKey(5) == ord('Q'):
        break

    if cv2.waitKey(5) == ord('r')  or cv2.waitKey(5) == ord('R'): # 이미지 초기화
        img = np.ones((100, 100, 1), dtype = np.uint8) * 255

    if cv2.waitKey(5) == 32: # 스페이스바

        resized_img = cv2.resize(img, (28, 28))
        resized_img = resized_img.reshape(1, 28, 28, 1).astype('float32') / 255.0
        pred = model.predict_step(resized_img)[0]

        # 출력은 2차원이므로 1차원으로 변경, 순서는 뒤집어준다
        pred = np.argsort(pred)[::-1]

        print("예측 1 : ", pred[0])
        print("예측 2 : ", pred[1])
        print("예측 3 : ", pred[2])
        print()
        

cv2.destroyAllWindows()

