from flask import Flask
from flask import request
import base64
import json
from ultralytics import YOLO
import cv2
import numpy as np

#학습결과 sign_language_yolo_model.pt를 이용해서 yolo 모델 형성
model = YOLO('c:/ai_project01/sign_language_yolo_model.pt')

app = Flask(__name__)

@app.route('/hello_rest_server', methods=['POST'])
def hello_world(): #put application's code here
    return '안녕 난 rest server야'



#post 방식의 param_rest_server url을 입력하면 함수가 호출되는 rest server의 메서드
@app.route("/param_rest_server", methods=['POST'])
def hello_rest2():
    #Rest Server로 전송한 name 파라미터의 값을 꺼내서 param_name 변수에 저장
    param_name = request.form.get('name', '입력값 없음')
    print("param_name=", param_name)
    #리턴값
    return "그래 너의 이름은 "+ param_name +" 이구나!!"

#post 방식의 image_test01 url을 입력하면 함수가 호출되는 rest server의 메서드
@app.route("/image_test01", methods=['POST'])
def image_send_test01():
    #request.get.json() : 스프링 컨트롤러에서 보낼 웹캠이미지를 리턴
    image = request.get_json()
    #스프링이 보낸 웹캠 이미지를 출력
    print("image=", image)
    #스프링 컨트롤러로 메시지 전송
    return "스프링이 보낸 이미지 잘 받았습니다!!";

#post 방식의 image_test02 url을 입력하면 함수가 호출되는 rest server의 메서드
@app.route("/image_test02", methods=['POST'])
def image_send_test02():
    #request.get_json() : 스프링 컨트롤러에서 보낼 웹캠이미지를 리턴
    image = request.get_json()
    print("image=", image)

    #image에서 key data의 value 리턴
    encoded_data = image.get("data")
    #image/jpeg;base64, 문자열 삭제
    encoded_data = encoded_data.replace("image/jpeg;base64,","")
    #encoded_data를 원래의 이미지로 저장
    decoded_data = base64.b64decode(encoded_data)

    #image.jpg 파일을 저장할 객체 f 생성
    with open('image.jpg', 'wb') as f:
        #decoded_data : 스프링 컨트롤러로부터 전송 받은 문자열을 이미지로 변화
        #파일로 저장
        f.write(decoded_data)

    return "스프링이 보낸 이미지 잘 저장했습니다";


#post 방식의 detect url을 입력하면 함수가 호출되는 rest server의 메서드
@app.route("/detect", methods=['POST'])
def detect_yolo():
    #request.get_json() : 스프링 컨트롤러에서 보낼 웹캠이미지를 리턴
    image = request.get_json()
    #스프링이 보낸 웹캠 이미지를 출력

    #print("="*100)
    #print("image=", image)
    #print("=" * 100)

    #image에서 key data의 value 리턴
    encoded_data = image.get("data")
    #image/jped;base64, 문자열 삭제
    encoded_data = encoded_data.replace("image/jpeg;base64,","")
    #encoded_data 를 원래의 이미지로 저장
    decoded_data = base64.b64decode(encoded_data)
    #print("=" * 100)
    #print("decoded_data =", decoded_data)
    #print("=" * 100)

    #np.fromstring(decoded_data, np.unit8) : decoded_data를
    #정수(np.uint8)가 저장된 1차원 배열로 변환해서 nparr에 저장
    nparr = np.fromstring(decoded_data, np.uint8)

    #cv2.imdecoded(nparr, cv2.IMREAD_COLOR) : 1차원 배열 nparr을 RGB형태의 3차원 배열로 변환
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #print("img=", img)

    #탐지 결과 리턴
    #imgsz : 학습했던 이미지 사이즈
    results = model.predict(img, imgsz=640)

    #Detect 결과 이미지를 detect_img에 저장
    #detect_img = results[0].plot()
    #Detect 결과 이미지를 파일명 detect_result.jpg로 저장
    #cv2.imwrite('detect_result.jpg', detect_img)
    #detect 결과를 저장할 배열
    box_result = []

    #detect 결과 하나를 r에 저장
    for r in results:
        #yolo 박스 정보 저장
        boxes = r.boxes
        for box in boxes:
            #box 좌표 저장
            left,top,right,bottom = box.xyxy[0]
            #클래스 타입 저장
            cls = int(box.cls)
            #모델 학습 레이블에서 cls(탐지결과) 번째의 클래스 이름 리턴
            cls_name = model.names[cls]
            #확률 저장
            conf = float(box.conf)

            #확률이 25% dltkddlaus
            if conf > 0.25:

                #yolo detect 결과 저장
                box_result.append({
                        "left":int(left), "top":int(top), "right":int(right),
                        "bottom":int(bottom), "cls": cls_name, "conf":float(conf) })

    #yolo 결과 리턴
    return json.dumps(box_result)

if __name__ == '__main__':
    app.run()