import numpy as np
import cv2
import tensorflow as tf 
import os
import time
import pickle

model = tf.keras.models.load_model('./model/korean_model_220706.h5')

with open('./main2_Korean_HandWriting/kr_label.pkl', 'rb') as f:
	data = pickle.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
img_size = (60, 60, 1)
# 필기체 구현 파라미터
drawing = False
pt_x, pt_y = None, None 

# 마우스 콜백 함수
def draw_lines(event, x, y, flags, param):

    global drawing, pt_x, pt_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt_x, pt_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt_x, pt_y), (x, y), color = BLACK, thickness = 3, lineType = cv2.LINE_AA)
            pt_x, pt_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt_x, pt_y), (x, y), color = BLACK, thickness = 3, lineType = cv2.LINE_AA)

img = np.ones(img_size, dtype = np.float32) * 255
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_lines)

while True:
    cv2.imshow('image', img)

    # 구현은 필기체 작성 -> 이미지를 추론 모델에 집어넣음 -> 예측 결과를 반환 하는 방식으로 이루어질 듯

    if cv2.waitKey(5) == ord('q') or cv2.waitKey(5) == ord('Q'):
        break

    if cv2.waitKey(5) == ord('r')  or cv2.waitKey(5) == ord('R'): # 이미지 초기화
        img = np.ones(img_size, dtype = np.float32) * 255
        # time.sleep(0.05)

    # 캡쳐 & 모델로 넘겨줌
    if cv2.waitKey(5) == 32: # 스페이스바
        resized_img = cv2.resize(img, (60, 60), interpolation = cv2.INTER_AREA) # 축소엔 INTER_AREA가 효과적이라고 함
        resized_img = resized_img.reshape(1, 60, 60, 1).astype('float32') / 255.0
        pred = model.predict_step(resized_img)[0]

        # 출력은 2차원이므로 1차원으로 변경, 순서는 뒤집어준다
        pred = np.argsort(pred)[::-1]
        
        print("예측 1 : ", data[pred[0]])
        print("예측 2 : ", data[pred[1]])
        print("예측 3 : ", data[pred[2]])

        # time.sleep(0.05)
        # 정렬 없이 예측 순위 높은 3개 출력하기
        # print(pred)
        # print(pred[0], pred[1], pred[2])
        
cv2.destroyAllWindows()

