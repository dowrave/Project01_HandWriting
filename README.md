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
