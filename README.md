## 🧏‍♀️ Sign Language Translator
- 개요 : 웹 캠으로 수어를 인식하고 텍스트로 출력하는 사이트
- 프로젝트 기간 : 2024년 7월 23일 ~ 2024년 8월 4일
- 프로젝트 인원 : 개인
- 사용 언어 및 개발 환경 : JAVA, SpringBoot, HTML, JSON, Eclipse, Python, Pycharm
- 세부 기능
  1. 웹 페이지에서 카메라로 수어를 인식하고 텍스트를 화면에 출력
> - HttpComponent를 이용해서 RestServer로 이미지를 전송, RestServer에서 탐지된 수어 결과를 가져옴
> - 실시간 웹캠 화면을 컨트롤러로 전송하기 위해 axios 라이브러리 사용
  2. 수어 탐지 시 확률 출력
> - 학습한 수어 데이터셋 중 확률이 가장 높은 인덱스를 찾아 JSON으로 변환한 후 리턴
  3. 앞의 수어를 통해 뒤의 수어를 예측하도록 LSTM 모델을 이용해서 수어 손동작 학습
> - 알파벳 수어 데이터를 직접 만들어 anaconda 가상환경에서 jupyter notebook을 통해 데이터 학습
