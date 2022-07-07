## OpenCV, Tensorflow를 이용해 마우스로 그린 글자를 분류하는 프로그램

- 작업기간 : `220624` ~ `220708`
    - `main2`는 코랩의 메모리 이슈 & 여러 어려움으로 인해 만족스러운 성능이 나오지는 못했습니다. 

## 실행 방법
- `main1` 폴더, `main2` 폴더 내의 `main1.py`(MNIST 0~9), `main2.py`(한글)를 실행
    - 흰 화면이 뜨면 그림판에 글씨 쓰듯이 글자를 그림(main1은 `0 ~ 9`의 숫자, main2는 한글 1글자)
    - spacebar를 (꾹) 누르면 모델이 추론한 상위 3개의 label이 출력
    - r을 (꾹) 누르면 흰 화면 리셋
    - q를 (꾹) 누르면 종료됨

## 완료
1. `mnist handwriting`에 관한 `cnn 모델`을 만들어 `OpenCV`에 그려진 글자를 인식시킨 뒤 분류 (완)
2. 한국어 글자 인식(모델, 데이터는 `Daily_log.md` 참조) (완, 성능은 불만족)

## 시행착오 내역
- `Daily_log.md`에 기록되어 있습니다.

## 참고 자료 및 논문, 프로그램
- 논문 1 : [GoogLenet 기반의 딥 러닝을 이용한 향상된 필기체 인식](https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201823955287871&oCn=JAKO201823955287871&dbt=JAKO&journal=NJOU00292001)
- 논문 2 : [한글 인식을 위한 CNN 기반의 간소화된 GoogLenet 연구](https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201630762630914&oCn=JAKO201630762630914&dbt=JAKO&journal=NJOU00431883)
- 사용 데이터 : [phd08](https://www.dropbox.com/s/69cwkkqt4m1xl55/phd08.alz?dl=0)
    - [참고 자료](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001293992)
- [phd08을 numpy 파일로 바꿔주는 프로그램](https://github.com/sungjunyoung/phd08-conversion)
    - `main2/phd08_to_npy.py` : 별도의 폴더에서 실행하여 넘파이 파일을 만들었으며, 전체가 아니라 일부만 필요했기 때문에 코드를 개선했음

## 느낀 점
1. 이렇게 길어질 줄 모르고 시작한 프로젝트인데, 어떤 어려움에 부딪혔을 때의 스트레스와 그걸 개선했을 때의 쾌감을 모두 느낄 수 있었다. <b>자발적으로 진행한 첫 프로젝트</b>이기도 하다.
2. 특히 main2의 경우는 처음엔 정확도가 `1e-5` ~`1e-4` 대를 벗어나지를 못했는데, `GoogLenet을 간소화한 모델`을 이용하고, 영상의 확대 / 축소 과정에서 `Interpolation`을 이용하면서 발생하는 값들을 보완하는 방법으로 `이진화`가 매우 효과적인 것을 확인할 수 있었다. 한 글자 씩 실험했기 때문에 통계적인 데이터가 있는 게 아닌 체감이지만, 확실하게 개선된 걸 느낄 수 있었다.
3. 아이디어를 네이버 사전 등에서 볼 수 있는 필기체를 인식해서 1글자씩 집어넣는 모델에서 얻었는데, 그게 얼마나 구현하기 힘든 건지도 몸소 깨달을 수 있었다. 
4. 어쩔 수 없이 포기한 게 2가지 있다.
    1. `opencv`의 `findContour`를 이용해서 그림판의 글씨 부분만 잘라낸 다음에 이를 추론 모델에 집어넣는 방식을 이용해보려 했다. 그러나 글씨만 딱 잘라내는 상황이 거의 나오지 않아서 포기했다.
    2. `phd08`에 있는 폰트가 9가지이고, 글씨 크기(3가지) 및 회전 각도(-3, 0, 3도)만을 따로 떼어내 각 글자마다 81개의 데이터를 이용해보려 했다. 그러나 한 케이스에 대한 데이터가 너무 적은 탓인지 모델 학습이 정상적으로 이루어지지 않았다. 이를 개선하기 위해 데이터 수를 3배로 늘렸으나 사실 코랩은 81개도 벅차서.. 그냥 기존의 `단일 폰트 50개`의 데이터를 이용한 모델을 남겨놨다.(`korean_model_220707.h5`)

