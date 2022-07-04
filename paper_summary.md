# [이 논문](https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201823955287871&oCn=JAKO201823955287871&dbt=JAKO&journal=NJOU00292001)
- PE92 기준으로 설명됨

- CNN의 한계 : 데이터가 적고 클래스가 많을수록 깊이가 낮은 네트워크에서는 학습이 잘 되지 않는다.
- GoogleNet, Resnet 등은 깊은 네트워크를 통해 이미지 인식에 우수한 성능을 보인다

# GoogleNet
- `299 * 299 * 3` 을 이용, 큰 이미지를 분류하는 네트워크
- 그러나 (이 논문에서는) PE92 : `60 * 60 * 1` 을 이용 -> 연산량을 줄이기 위해 경량화함

## GoogleNet 설명
1. 보조 분류기 - 중간 단에 배치, 이미지 분류를 도우나 이 실험에서의 성능 향상이 눈에 띄지 않아 제외
2. Stem 영역 
- 구성
```
Convolution(7*7 / 2)
MaxPool(3*3 / 2)
Convolution(3*3 / 1)
MaxPool(3*3 / 2)
```
- 추후 적용할 Inception 모듈을 바로 적용하면 분류 효과가 떨어져 여기서 Feature Map을 생성함
- 이 논문에서는 아래처럼 개선함
```
Convolution (5 * 5 / 1)
MaxPool(3 * 3 / 2)
```
3. Inception 모듈
- `1, 3, 5` 3개의 필터를 사용해 데이터의 인접 정보를 모은다 -> 특징 맵들을 합쳐 다음 레이어의 입력으로 전달한다.
- 효과 : 가 / 갸, 뮛 / 뭿 같이 유사 클래스의 분류 성능을 단일 3*3 필터를 통해 네트워크를 쌓는 것보다 좋음
    - 채널의 부피가 늘어나기 때문에 3, 5 필터 앞에 1 필터를 붙여 채널의 크기를 조절함

## 전체 필기체 네트워크의 구성
- 이미지 전처리 : 60 * 60 -> 선형 보간법으로 이미지 고정
1. Convoluton & Max_Pool 사용
    - Gradient Exploding & Vanishing을 막기 위해 `BatchNormalization`을 넣고 활성화 함수는 `ReLU`를 넣음
    - 전체 구조는 `Convolution - Batch_Norm - ReLU - Max_Pool`
2. Inception 모듈
    - 채널은 유지
    - Max_Pool 이전에 Inception 모듈이 위치하면 Pooling 레이어의 채널 수를 2배로 늘려줌(정보 손실 완화)
    - Inception 모듈 연산 결과 `7 * 7 * 512`의 채널이 만들어지며 이들 데이터의 평균을 구한 뒤 FC를 통해 분류할 수 있도록 `1*1*512`의 노드로 변형 후 2350개의 FC를 통해 분류한다.
3. 오차함수
- MSE : 기울기 값이 0에 가깝게 나와 수렴이 느려짐
- 그래서 CE를 쓴다.
4. 학습률 변화
- 학습시간이 길어지면 학습률 향상 속도가 느려짐(Adam의 디폴트 : 1e-3) 
- 가장 성능이 좋았던 건 30에포크마다 학습률을 `1e-3 -> 5e-4 -> 1e-4`로 낮춘 `Cliff Drop` 이었음