# 한글 단일 글자 손글씨 분류 프로그램

## 개요
1. 작업기간 : `220624 ~ 220708`
2. 사용 라이브러리 / 프레임워크 : `Tensorflow`, `cv2(OpenCV)`, `sklearn`, `numpy`, `pandas`, `matplotlib` 등
3. 실사용보다는 <b>생각한 것을 구현할 수 있는가?</b>에 중점을 둔 프로젝트입니다. 

## 내용
1. `main1` 폴더 
    - 구글 코랩 환경에서 60000개의 MNIST 데이터셋 `28*28`을 이용해 텐서플로우로 간단한 CNN 모델을 학습시킨 뒤 로컬에 저장했습니다.
    - 로컬에서는 OpenCV을 이용해 `100*100*1`에 손글씨를 그릴 도화지와 선을 그리는 함수를 구현하여 손글씨 이미지를 `28*28*1`로 리사이징하여 텐서플로우 모델에 전달해 추론시켰습니다.
    - `OpenCV에서 그림을 그린다 -> 이를 텐서플로우에 전달해 추론한다` 를 테스트하기 위한 부분입니다.

2. `main2` 폴더
    - 구글 코랩 환경에서 `phd08`이라는 한글 단일 글씨 데이터셋을 이용, `60*60*1`의 OpenCV 이미지를 2350개의 Label로 구분하는 GoogLeNet을 간소화한 모델을 생성한 뒤 로컬에 저장했습니다. 
    - 로컬에서는 위와 같이 `100*100*1`의 이미지를 `60*60*1`로 리사이징 및 이진화하여 추론했습니다.

<details>
<summary><h2>실행 방법</h2></summary>
- `main1` 폴더의  `main1.py`(숫자 0~9) or `main2` 폴더의 `main2.py`(단일 한글 글자)를 실행
    - 흰 화면이 뜨면 그림판에 글씨를 쓰듯 숫자나 문자를 그린다.
    - spacebar를 (꾹) 누르면 모델이 추론한 상위 3개의 label이 출력
    - r을 (꾹) 누르면 흰 화면으로 초기화
    - q를 (꾹) 누르면 종료됨
</details>

## 실행 결과
- 검증, 테스트 성능이 99.5%까지 나왔는데, 손글씨에 대해서는 만족스러운 성능이 나오지 못했습니다. 자세한 내용은 후기에 작성해놓았습니다. 
- 직접 손글씨를 그리며 추론 성능을 봤기 때문에 수치적인 성능을 볼 수 없었다는 점도 아쉬웠습니다.
- (성공 결과) (실패 결과)

<details>
<summary><h2>아쉬운 점 & 분석</h2></summary>
대충 넣을 글자들

</details>

## 참고 자료 및 논문, 프로그램
- 논문 1 : [GoogLenet 기반의 딥 러닝을 이용한 향상된 필기체 인식](https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201823955287871&oCn=JAKO201823955287871&dbt=JAKO&journal=NJOU00292001)
- 논문 2 : [한글 인식을 위한 CNN 기반의 간소화된 GoogLenet 연구](https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201630762630914&oCn=JAKO201630762630914&dbt=JAKO&journal=NJOU00431883)
- 사용 데이터 : [phd08](https://www.dropbox.com/s/69cwkkqt4m1xl55/phd08.alz?dl=0)
    - [참고 자료](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001293992)
- [phd08을 numpy 파일로 바꿔주는 프로그램](https://github.com/sungjunyoung/phd08-conversion)
    - `main2/phd08_to_npy.py` : 별도의 폴더에서 실행하여 넘파이 파일을 만들었으며, 전체가 아니라 일부만 필요했기 때문에 코드를 개선했음
