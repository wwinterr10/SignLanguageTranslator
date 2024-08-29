import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

from ultralytics import YOLO

#학습한 YOLO 모델 읽어서 리턴
model = YOLO("c:/ai_project01/sign_language_yolo_model.pt")

#mp.solutions.hands : 화면에서 손과 소가락 관절 위치 정보를 탐지 객체 리턴
mp_hands = mp.solutions.hands

#cv2.VideoCapture(0) : 웹캠의 화면을 가져올 객체를 생성해서 리턴
cap = cv2.VideoCapture(0)

#with mp_hands.Hands : 여기부터 화면에서 손과 손가락 위치 탐지
with mp_hands.Hands() as hands:

    #cap.isOpened() : 웹캠이 정상적으로 동작하면 True, 웹캠이 정상적으로 동작하지 못하면 False 리턴
    while cap.isOpened()==True: #웹캠이 정상적으로 동작하는 동안 반복

        #cap.read() : 웹캠의 화면을 가져옴

        #웹캠의 화면을 정상적으로 가져오면 success에 True가 저장
        #웹캠의 화면을 가져오는데 실패하면 success에 False가 저장

        #웹캠이 가져온 화면은 image에 저장
        success, image = cap.read()

        #cv2.flip(image, 1) : 화면 좌우 대칭
        #cv2.flip(image, 0) : 화면 상하 대칭
        image = cv2.flip(image, 1)

        if success==False: #웹캠의 화면을 가져오는데 실패하면
            continue #반복문의 다음으로

        #cv2.cvtColor(image, cv2.COLOR_BGR2RGB) : 웹캠 이미지 image를 BGR에서 RGB로 변환
        #hands.process() : 웹캠에서 손과 손가락 관절 위치 탐지해서 리턴
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        #results.multi_hand_landmarks : 웹캠 화면에서 손이 없으면 None이 리턴
        #results.multi_hand_landmarks != None : 화면에 손이 존재하면
        if results.multi_hand_landmarks != None:
            #웹캠 이미지에 글자 합성
            #cv2.putText(
            #    image,
            #    text="Detect Hand!!",
            #    org=(300, 50),
            #    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #    fontScale=1,
            #    color=(0, 0, 255),
            #    thickness=2
            # )

            #cv2.cvtColor(image, cv2.COLOR_BGR2RGB) : 웹캠 이미지 image를 BGR에서 RGB로 변환
            #model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), stream=True) : YOLO모델을 이용해서 수어 탐지
            #stream=True : 실시간 이미지

            #탐지한 수어 결과는 results에 저장
            results = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), imgsz=800)

            #results에서 탐지한 결과 하나씩 r에 저장
            for r in results:
                #r.boxes : 탐지한 수어 좌표들을 리턴
                boxes = r.boxes

                for box in boxes:
                    #box.xyxy[0] : 수어의 x1,y1,x2,y2 좌표 리턴
                    x1, y1, x2, y2 = box.xyxy[0]

                    #int(x1) : x1 정수로 변환
                    x1 = int(x1)
                    #int(y1) : y1 정수로 변환
                    y1 = int(y1)
                    #int(x2) : x2 정수로 변환
                    x2 = int(x2)
                    #int(y2) : y2 정수로 변환
                    y2 = int(y2)

                    #화면에 박스를 그림
                    cv2. rectangle(image,       #박스를 그릴 이미지
                                   (x1, y1),    #박스의 x1, y1 좌표
                                   (x2, y2),    #박스의 x2, y2 좌표
                                   (0, 255, 0), #박스 색 BGR순으로 녹색
                                   3            #박스 선 두께
                                   )

                    #box.cls : 탐지결과 0->A, 1->B, 2->C .... 25->Z
                    print("box.cls=", box.cls)

                    #탐지 결과를 정수로 변환
                    cls = int(box.cls[0])

                    #model.names : 모델 학습 레이블 출력
                    print("model.names=", model.names)

                    #모델 학습 레이블에서 cls (탐지결과) 번째의 클래스 이름 리턴
                    cls_name = model.names[cls]

                    #boix.conf[0] : 탐지된 confidence (IOU*확률) 출력
                    print("box.conf[0]=", box.conf[0])

                    conf_score = float(box.conf[0])
                    #box.conf[0] : 탐지된 confidence (IOU*확률) 출력
                    #round(box.conf[0], 3) : box.conf[0]를 소숫점 아래 2자리 반올림
                    confidence = round(conf_score, 2)

                    #str(confidence) : confidence를 문자열로 변환
                    yolo_text = cls_name + ":" + str(confidence)

                    #웹캠 이미지에 글자 합성
                    cv2.putText(
                        image, #웹캠 이미지
                        text=yolo_text, #합성할 글자
                        org=(x1, y1), #글자를 합성할 가로 세로 위치
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, #합성할 글자 폰트
                        fontScale=1, #합성할 글자 크기
                        color=(0, 255, 0), #합성할 글자 색
                        thickness=2 #핪어할 글자 두께
                    )

        #cv2.imshow (윈도우창 타이틀, 윈도우창에 출력할 이미지) : 웹캠 화면을 화면에 출력
        cv2.imshow('webcam_window01', image)
        
        #cv2.waitKey(1) : 사용자가 키보드 입력하도록 1초 기다림
        #                 기다리는 시간동안 사용자가 키보드 입력을 하면 입력한 키보드값을 리턴
        #                 기다리는 시간동안 사용자가 입력한 키보드가 없으면 None 리턴
        
        if cv2.waitKey(1) == ord('q') #키보드 입력이 q이면
            
            #opencv의 BGR로 되어있는 이미지를 RGB로 변환
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            #image 변수에 저장된 웹캠 이미지를 cam_img.jpg 파일로 저장
            #plt.imsave("cam_img.jpg", image)

            break #반복문 종료
            
#웹캠의 화면 그만 가져오도록 설정
cap.release()