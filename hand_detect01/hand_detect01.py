import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
#mp solutions.hands : 화면에서 손과 손가락 관절 위치 정보를 탐지 객체 리턴
mp_hands = mp.solutions.hands

#인식한 손의 key point를 그릴 객체
mp_drawing = mp.solutions.drawing_utils

# cv2.VideoCapture(0) : 웹캠의 화면을 가져올 객체를 생성해서 리턴
cap = cv2.VideoCapture(0)

#with mp_hands.Hands : 여기부터 화면에서 손과 손가락 위치 탐지
with mp_hands.Hands() as hands:

    # cap.isOpened() : 웹캠이 정상적으로 동작하면 True, 웹캠이 정상적으로 동작하지 못하면 False 리턴
    while cap.isOpened()==True: #웹캠이 정상적으로 동작하는 동안 반복

        # cap.read() : 웹캠 화면을 가져옴
        # 웹캠의 화면을 정상적으로 가져오면 success에 True 가 저장
        # 웹캠의 화면을 가져오는데 실패하면 success에 False가 저장
        # 웹캠의 가져온 화면을 image에 저장
        success, image = cap.read()

        #cv2.flip(image, 1) :웹캠 이미지를 좌우 반전
        image = cv2.flip(image, 1) #1은 좌우 반전, 0은 상하 반전입니다

        if success == False : # 웹캠의 화면을 가져오는데 실패하면
            continue # 반복문의 다음으로

        #cv2.cvtColor(image, cv2.COLOR_BGR2RGB) : 웹캠 이미지 image를 BGR에서 RGB로 변환
        #hands.process () : 웹캠에서 손과 손가락 관절 위치 탐지해서 리턴
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        #results.multi_hand_landmarks : 웹캠 화면에서 손이 없으면 None이 리턴
        #results.multi_hand_landmarks != None : 화면에 손이 존재하면
        if results.multi_hand_landmarks != None:
            #웹캠 이미지에 글자 합성
            cv2.putText(
                image, #웹캠 이미지
                text="Detect Hand!!", #합성할 글자
                org=(300, 50), #글자를 합성할 가로 세로 위치
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, #합성할 글자 폰트
                fontScale=1, #합성할 글자 크기
                color=(0, 0, 255), #합성할 글자색
                thickness=2 #합성할 글자 두께
            )

            #results.multi_hand_landmarks : 탐지한 손의 keypoint들이 저장
            #for hand_landmarks in results.multi_hand_landmarks : 탐지한 손의 keypoint 순서대로 1개씩 hand_landmarks에 저장하고 반복문 실행

            for hand_landmarks in results.multi_hand_landmarks:
                #탐지한 손의 keypoint를 화면에 그림
                mp_drawing.draw_landmarks(
                    image, #keypoint를 그릴 이미지 (웹캠화면)
                    hand_landmarks, #keypoint 좌표
                    mp_hands.HAND_CONNECTIONS, #탐지한 손의 keypoint를 선으로 연결함
                    #keypoint를 표시할 원의 모양
                    #color=(0, 255, 0) : 녹색 thickness=2 : 선두께 circle_radius=2 : 원반지름
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )

                #hand_landmarks.landmark[12]의 y좌표
                #hand_landmarks.landmark[9].y의 y좌표
                if hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y :
                    #웹캠 이미지에 글자 합성
                    cv2.putText(
                        image, #웹캠 이미지
                        text="Open!!", #합성할 이미지
                        org=(300, 100), #글자를 합성할 가로 세로 위치
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, #합성할 글자 폰트
                        fontScale=1, #합성할 글자 크기
                        color=(0, 0, 255), #합성할 글자색
                        thickness=2 #합성할 글자 두께
                    )

                else:
                    #웹캠 이미지에 글자 합성
                    cv2.putText(
                        image,  # 웹캠 이미지
                        text="Close!!",  # 합성할 이미지
                        org=(300, 100),  # 글자를 합성할 가로 세로 위치
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 합성할 글자 폰트
                        fontScale=1,  # 합성할 글자 크기
                        color=(0, 0, 255),  # 합성할 글자색
                        thickness=2  # 합성할 글자 두께
                    )


        # cv2.imshow (윈도우창 타이틀, 윈도우창에 출력할 이미지) : 웹캠 화면을 윈도우에 출력
        cv2.imshow('webcam_window01', image)

        # cv2.waitKey(1) : 사용자가 키보드 입력하도록 1초 기다림
        #                  기다리는 시간동안 사용자가 키보드를 입력을 하면 입력한 키보드값을 리턴
        #                  기다리는 동안 사용자가 입력한 키보드가 없으면 None 리턴

        if cv2.waitKey(1) == ord('q'):  # 키보드 입력이 q이면

            #opencv의 BGR로 되어있는 이미지를 RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #image 변수에 저장된 웹캠 이미지를 cam_img.jpg 파일로 저장
            plt.imsave("cam_img.jpg", image)

            break  # 반복문 종료

    # 웹캠의 화면 그만 가져오도록 설정
    cap.release()