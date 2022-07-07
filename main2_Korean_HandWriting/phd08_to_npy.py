import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import re
# from scipy.misc import imresize # 없어진 함수
# cmd에서 다음 명령어로 실행 : python phd08_to_npy.py --data_dir=phd08_inputs --width=60 --height=60 --batch_size=1

# (수정) 갯수 제한 변수
limited_num = True # 정규식 만족하는 조건들만 고르기 위한 변수
counts = 100

# 실제로는 counts = 50으로 놓고 50개만 뽑고 다음 파일로 넘어가게끔 사용함
# 220707 : 글자 별로 조건에 맞는 데이터 81개(9 * 3 * 3)만 뽑아서 씀
# a = re.compile("[A-Z0-9]+_[0-2]_0_2_1_[0-2]") # FO_FS_0_2_1_SL 형태
# target_checker = False

def parse_args():
    desc = "phd08 한글 텍스트 데이터를 numpy array 로 바로 변환 가능한 npy 파일 형태로 바꿔좁니다."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir', type=str, default='phd08_sample',
                        help='phd08 한글 데이터가 존재하는 디렉토리', required=True)
    parser.add_argument('--one_hot', action='store_true', help='라벨 원 핫 벡터 여부')
    parser.add_argument('--width', type=int, default=28,
                        help='저장할때의 가로 어레이 사이즈', required=False)
    parser.add_argument('--height', type=int, default=28,
                        help='저장할때의 세로 어레이 사이즈', required=False)
    parser.add_argument('--gaussian_sigma', type=float, default=.3,
                        help='가우시안 필터 적용 시 시그마 값', required=False)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='몇개의 글자씩 합쳐서 저장할 것인지 (max 10)', required=False)

    return parser.parse_args()


def font_start_checker(line):
    if not line.strip(): # line.strip() : 각 줄의 사이드를 없애고 남은 내용
        return True # 
    else:
        return False



# if else문이라 continue가 필요한가 싶긴 한데...
def txt_to_npy(all_file_count, index, data, labels,
               file_full_path, width, height, sigma, is_one_hot):
    index -= 1
    global target_checker
    with open(file_full_path, 'r') as lines:
        font_counter = 0
        not_data_checker = 0
        font_array = []
        real_data_counter = 0
        for line in lines:
            if font_start_checker(line):  # endl / 어떤 줄의 남은 내용이 없으면 True, 있으면 False임
                                          # 즉 데이터를 다 집어넣고 빈 칸이 나오면 여태까지 담은 array를 정리하는 거임
                not_data_checker = 0
                target_checker = False
                font_counter += 1
                real_data_counter = 0

                if len(font_array) == 0:
                    continue
                # data
                font_blurred_array = gaussian_filter(font_array, sigma=sigma)
                # font_blurred_array = imresize(font_blurred_array, [height, width])
                # numpy.array(Image.fromarray(arr).resize())
                font_blurred_array = np.array(Image.fromarray(font_blurred_array).resize((width, height)))
                font_blurred_array = font_blurred_array.astype(np.float32)

                data.append(font_blurred_array)

                # labels
                if is_one_hot == True:
                    label = np.zeros(shape=(all_file_count))
                    label[index] = 1
                else:
                    label = index
                labels.append(label)

                font_array = []

                # 100개가 채워지면 다음 파일로 넘어감 - 220707 : 없애도 될 것 같긴 함
                if len(data) == counts and len(labels) == counts:
                    break

                continue
        
            else:  # not endl
                not_data_checker += 1
                if not_data_checker == 1:  # font name
                    # 여기서 설정할 수 있을 거 같음
                    # FO_FS_CP_RE_TH_SL 형태고
                    # FO_FS_0_2_1_SL (폰트 종류 9개/ 크기 3개 / 복사 0 / 해상도 280 / 이진 임계치 180 / 기울기 3개) 이렇게만 쓸 예정
                    if a.match(line):
                        target_checker = True
                    else: 
                        continue
                elif target_checker: # target_checker가 켜진 경우 글자들을 넣어준다
                    if not_data_checker == 2:  # font size
                        arr_height = int(line.split(' ')[0])
                        arr_width = int(line.split(' ')[1])
                        font_array = np.zeros(shape=(arr_height, arr_width))
                        continue
                    else:  # this is real data
                        font_array[real_data_counter] = list(map(int, line.strip()))
                        real_data_counter += 1
                        continue
                else:
                    # 각 데이터별 2번째 줄부터이면서 target_checker가 꺼졌고 line에 데이터가 있는 경우
                    # line에 들어오는 데이터가 없어질 때까지 계속 continue하면 됨
                    continue
                

    return data, labels


def main():
    args = parse_args()
    if args is None:
        exit()

    # 존재유무 체크
    if not os.path.exists(args.data_dir):
        print("ERROR::" + args.data_dir, " 는 존재하지 않는 폴더입니다.")
        exit()

    save_dir = 'phd08_npy_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = []
    labels = []
    # 쪼개서 저장할 사이즈
    batch_size = args.batch_size

    print(args)
    # 전체 파일 갯수 체크
    all_file_count = 0
    for _, _, files in os.walk(args.data_dir):
        for file in files:
            if file[0] == '.':
                continue
            all_file_count += 1


    for _, _, files in os.walk(args.data_dir):
        index = 0
        for file in files:
            if file[0] == '.':
                continue
            index += 1
            print("INFO:: converting " + file + "...")
            data, labels = txt_to_npy(all_file_count, index, data, labels,
                                      args.data_dir + '/' + file, args.width, args.height, args.gaussian_sigma,
                                      args.one_hot)

            # 리스트로 만들고 마지막에 np.array를 씌우는 방식임
            if index % batch_size == 0:
                data = np.array(data, dtype=np.float32)
                labels = np.array(labels)
                out_data = save_dir + '/phd08_data_' + str(int(index / batch_size))
                out_labels = save_dir + '/phd08_labels_' + str(int(index / batch_size))
                np.save(out_data, data)
                np.save(out_labels, labels)
                data = []
                labels = []
                print("  FILE_SAVED:: filename : " + save_dir + '/phd08_data_' + str(int(index / batch_size)))

    print("INFO:: all files converted to npy file, results in phd08_npy_results")


if __name__ == '__main__':
    main()
