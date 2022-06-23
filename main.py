import numpy as np
import cv2
import tensorflow as tf 
# 드래그를 구현하려면 LBUTTON_DOWN 때 별도의 파라미터를 on/off 식으로 바꿔주면 됨
drawing = False
pt_x, pt_y = None, None # cv2.circle은 그리는 선이 끊기는 듯함
                        # 그래서 cv2.line의 끝 점을 설정하기 위해 별도의 변수를 둠

# 모델 불러오기
model = tf.keras.models.load_model('my_model.h5')

# 마우스 그리기
def draw_lines(event, x, y, flags, param):

    global drawing, pt_x, pt_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt_x, pt_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt_x, pt_y), (x, y), color = (0, 0, 0), thickness = 2, lineType = cv2.LINE_AA)
            pt_x, pt_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt_x, pt_y), (x, y), color = (0, 0, 0), thickness = 2, lineType = cv2.LINE_AA)

# 일단 MNIST 필기체 인식부터 시작. 바탕은 흰색으로 ㄱㄱ
img = np.ones((100, 100, 1), dtype = np.uint8) * 255
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_lines)

# model = tf.keras.models.load_model('my_model.tf')

while True:
    cv2.imshow('image', img)

    # 구현은 필기체 작성 -> 이미지를 추론 모델에 집어넣음 -> 예측 결과를 반환 하는 방식으로 이루어질 듯

    if cv2.waitKey(5) == ord('q') or cv2.waitKey(5) == ord('Q'):
        break

    if cv2.waitKey(5) == ord('r')  or cv2.waitKey(5) == ord('R'): # 이미지 초기화
        img = np.ones((100, 100, 1), dtype = np.uint8) * 255
        # time.sleep(0.05)

    # 여기가 캡쳐해서 모델로 넘겨주는 부분임
    if cv2.waitKey(5) == 32: # 스페이스바
        # 여기서 그림판에 그려진 글씨 자체의 array를 텐서플로우에 전달하면 됨
        # 근데 이미지 가공이 필요함 : 28 * 28을 맞춰줘야 하므로
        resized_img = cv2.resize(img, (28, 28))
        resized_img = resized_img.reshape(28, 28, 1)
        # print(resized_img.shape)
        print(model.predict_step(resized_img))

cv2.destroyAllWindows()

