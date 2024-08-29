import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

import pandas as pd
import numpy.linalg as LA

#예측값과 문자열
gesture = {
    0:'FIRST', 1:'ONE', 2:'TWO', 3:'THREE', 4:'FOUR', 5:'FIVE', 6:'SIX', 7:'ROCK', 8:'SPIDERMAN', 9:'YEACH', 10:'OK'
}

#csv 파일의 내용 출력시 모든 컬럼 출력하도록 설정
pd.set_option('display.max_columns', None)
#csv 파일의 내용 출력시 한줄에 모든 컬럼 출력하도록 설정
pd.set_option('display.expand_frame_repr', False)

#pd.read_csv() : csv 파일을 읽어서 내용을 리턴

#pd.read_csv(
#             #읽을 파일 경로
#             'c:/ai_project01/gesture_train.csv',
#
#             #파일의 첫번째 줄이 컬럼 명인지 여부
#             header=None #파일 첫번째 줄이 컬럼 이름 아님
#             )
gesture_df = pd.read_csv('c:/ai_project01/gesture_train.csv', header=None)
#print("gesture_df=", gesture_df)
#print("=" *100)

#gesture_df.iloc[줄인덱스, 칸인덱스]

# : 모든 줄
# : -1 0번째 칸부터 마지만 칸 미만까지 리턴
#gesture_df.iloc[:,:-1] : gesture_df의 모든줄, 마지막 칸 앞까지 리턴
angle=gesture_df.iloc[:,:-1]

#print("angle=",angle)
#print("=" *100)

##type(angle) : angle 변수의 타입 조회 여러 컬럼이 저장된 DataFrame
#print("type(angle)=", type(angle))

##angle.values : angle을 numpy 배열로 변환
#print("angle.values", angle.values)
##type(angle.values) : angle.values 변수의 타입 조회 numpy 배열 ndarray
##실수가 저장되어있기 때문에 타입은 8바이트 실수 (float64)
#print("type(angle.values)=", type(angle.values))

##angle.values.astype(np.float32) : angle.values의 타입을 32바이트 실수 np.float32로 변환
##astype(np.float32) : 배열 타입을 float32로 변환시키는 함수
angle_arr = angle.values.astype(np.float32)

#print("angle_arr=", angle_arr)
#print("=" *100)

#gesture_df,iloc[줄인덱스, 칸인덱스]

# : 모든줄
# : -1 마지막 칸 리턴
#gesture_df.iloc[:,-1] : gesture_df의 모든줄, 마지막 칸 리턴
label = gesture_df.iloc[:,-1]
#print("label=", label)
#print("=" *100)

##type(label) : label 변수의 타입 조회 1개 컬럼이 저장된 Series
#print("type(label)=", type(label))

#label.values : label을 numpy 배열로 변환
#print("label.values", label.values)
#type(label.values) : label.values 변수의 타입 조회 numpy 배열 ndarray
#실수가 저장되어있기 때문에 배열의 타입은 8바이트 실수(float64)
#print("type(label.values)=", type(label.values))

#label.values.astype(np.float32) : label.values의 타입을 32바이트 실수 np.float32로 변환
#astype(np.float32) : 배열 타입을 float32로 변환시키는 함수
label_arr = label.values.astype(np.float32)

#print("label_arr=", label_arr)
#print("=" *100)

#cv2.ml.KNearest_create() : KNN 객체 생성
knn = cv2.ml.KNearest_create()

#새로운 데이터와 거리를 구할 데이터 준비
#cv2.ROW_SAMPLE : 하나의 데이터가 한 행으로 구성됨
#cv2.COL_SAMPLE : 하나의 데이터가 한 열로 구성됨

#knn.train(angle_arr => 거리를 계산 할 keypoint 각도, cv2.ml.ROW_SAMPLE, label => 각도의 손모양)
knn.train(angle_arr, cv2.ml.ROW_SAMPLE, label_arr)

#창의 크기를 수정 가능하도록 설정 cv2.WINDOW_NORMAL
#cv2.namedWindow(윈도우 창 타이틀, flags=cv2.WINDOW_NORMAL)
cv2.namedWindow(winname='webcam_window01', flags=cv2.WINDOW_NORMAL)

#창의 크기를 수정
#cv2.namedWindow(윈도우 창 타이틀, width=윈도우 크기 가로, height=윈도우 크기 세로)
cv2.resizeWindow(winname='webcam_window01', width=1024, height=800)

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
        # 웹캠의 화면을 정상적으로 가져오면 success에 True가 저장
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
            #cv2.putText(
            #   image, #웹캠 이미지
            #   text="Detect Hand!!", #합성할 글자
            #   org=(300, 50), #글자를 합성할 가로 세로 위치
            #   fontFace=cv2.FONT_HERSHEY_SIMPLEX, #합성할 글자 폰트
            #   fontScale=1, #합성할 글자 크기
            #   color=(0, 0, 255), #합성할 글자색
            #   thickness=2 #합성할 글자 두께
            #)

            #results.multi_hand_landmarks : 탐지한 손의 keypoint들이 저장
            #for hand_landmarks in results.multi_hand_landmarks : 탐지한 손의 keypoint 순서대로 1개씩 hand_landmarks에 저장하고 반복문 실행

            for hand_landmarks in results.multi_hand_landmarks:
                #np.zeros((21, 3)) : 21줄 3칸의 0으로 초기화된 배열 생성
                joint = np.zeros((21, 3))

                #hand_landmarks.landmark : 손의 keypoint 좌표 리턴
                # j : keypoint의 index
                #lm : keypoint 좌표
                for j, lm in enumerate(hand_landmarks.landmark):
                    #j : keypoint index
                    #print("j=",j)
                    #lm : keypoint 좌표
                    #print("lm=",lm)
                    #lm.x : keypoint의 x좌표
                    #print("lm.x=",lm.x)

                    #lm.y : keypoint의 y좌표
                    #print("lm.y=",lm.y)

                    #lm.z : keypoint의 z좌표
                    #print("lm.z=",lm.z)
                    #각도를 구하기 위해 keypoint의 x,y,z 좌표를
                    #joint배열 j번째 행에 대입
                    joint[j] = [lm.x, lm.y, lm.z]
                    #print("=" *100)

                #print("joint=", joint)

                    #v1에서 v2를 빼서 차를 계산
                                #배열의 행 인덱스                  #배열의 열 인덱스
                                                                #x,y,z
                    v1 = joint[ [0, 1, 2, 3, 0, 5, 6, 7, 0, 9,  10, 11,  0, 13, 14, 15,  0, 17, 18, 19],    :]
                    v2 = joint[ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],    :]
                    #v2에서 v1빼기
                    v = v2 - v1
                    #print("=" *100)
                    #print("v", v)
                    #print("=" *100)

                    # v를 정규화시킴
                    #정규화 결과는 1차원 배열
                    v_normal = LA.norm(v, axis=1)
                    #print("=" * 100)
                    #print("v_normal=", v_normal)
                    #print("=" * 100)

                    #v와 연산하기 위해서 v_normal을 2차원 배열로 변환
                    #[:, np.newaxis]
                    #                  : => 모든 행
                    #                  np.newaxis => 차원 추가
                    #모든 행의 차원을 추가해서 1차원 배열 v_normal을 2차원 배열 v_normal2로 변환
                    v_normal2 = v_normal[:, np.newaxis]
                    #print("v_normal2=", v_normal2)

                    #v를 v_normal2로 나눠서 거리를 정규화시킴
                    v2 = v / v_normal2
                    #print("=" * 100)
                    #print("v2", v2)
                    #print("=" * 100)

                    print("joint", joint)

                    #a,b 배열의 곱을 계산
                    a = v2[[0, 1, 2, 4, 5, 6, 8,  9, 10, 12, 13, 14, 16, 17, 18], :]
                    b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                    ein = np.einsum('ij,ij->i', a, b)
                    #print("=" *100)
                    #행렬의 곱 조회
                    #print("ein=", ein)
                    #print("=" *100)

                    #코사인값 계산 (라디안)
                    #np.arccos(0,5) #1.0417975511965979 -> pi/3
                    radian = np.arccos(ein)
                    #print("radian=", radian)

                    #코사인값을 각도로 변환
                    #np.degrees(1,0471975511965979) : 60도
                    angle = np.degrees(radian)
                    #print("angle=",angle)

                    #angle을 numpy 배열로 전환
                    data = np.array([angle], dtype=np.float32)
                    #print("data=", data)

                    #knn.findNearest(data, 3) : 현재 손의 각도 data와 KNN에 저장된 angle_arr과 거리를 계산하고
                    #가장 거리가 가까운 3개 조회
                    #거리가 가까운 손모양 결과 (실수로 리턴) retval
                    #거리가 가까운 손모양 결과 (배열로 리턴) results
                    #거리가 가까운 3개의 손모양 결과
                    #가장 가까운 거리 3개
                    retval, results, neighbours, dist = knn.findNearest(data, 3)

                    print("=" * 100)
                    # 거리가 가까운 손모양 결과 (실수로 리턴)
                    print("retval=", retval)
                    print("=" * 100)
                    # 거리가 가까운 손모양 결과 (배열로 리턴)
                    print("results=", results)
                    print("=" * 100)
                    # 거리가 가까운 3개의 손모양 결과
                    print("neighbours=", neighbours)
                    print("=" * 100)
                    # 가장 가까운 거리 3개
                    print("dist=", dist)
                    print("=" * 100)

                # int(retval) : 예측값을 정수로 변환
                idx = int(retval)
                print("idx=", idx)
                # image.shape[0] : 이미지 세로 사이즈
                print("image. shape[0]=", image.shape[0])
                # image.shape[1] : 이미지 가로 사이즈
                print("image. shape[1]=", image.shape[1])
                # hand_Landmarks. Landmark[0].x : 손위치 X좌표 (8~1)
                print("hand_landmarks. landmark[0].x=", hand_landmarks.landmark[0].x)
                # hand_Landmarks. Landmark[0].y: 손위치 y좌표 (8~1)
                print("hand_landmarks. landmark[0].y=", hand_landmarks.landmark[0].y)

                # 예측값이 0~10 사이이면
                if 0 <= idx <= 10:

                    cv2.putText(image,
                            text=gesture[idx],  #예측값
                            org=(
                                int(hand_landmarks.landmark[0].x * image.shape[1]),  # 손위치 x좌표
                                int(hand_landmarks.landmark[0].y * image.shape[0])   # 손위치 y좌표
                                ),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,  #글자체
                            fontScale=1,                        #글자크기
                            color=(0, 0, 255),                  #글자색
                            thickness=2                         #폰트 두께
                            )

                # 탐지한 손의 keypoint를 화면에 그림
                mp_drawing.draw_landmarks(
                    image,  # keypoint를 그릴 이미지 (웹캠화면)
                    hand_landmarks,  # keypoint 좌표
                    mp_hands.HAND_CONNECTIONS,  # 탐지한 손의 keypoint를 선으로 연결함
                    # keypoint를 표시할 원의 모양
                    # color=(0, 255, 0) : 녹색 thickness=2 : 선두께 circle_radius=2 : 원반지름
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )

                #hand_landmarks.landmark[12]의 y좌표
                #hand_landmarks.landmark[9].y의 y좌표
                #if hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y :
                #    #웹캠 이미지에 글자 합성
                #    cv2.putText(
                #        image, #웹캠 이미지
                #        text="Open!!", #합성할 이미지
                #        org=(300, 100), #글자를 합성할 가로 세로 위치
                #        fontFace=cv2.FONT_HERSHEY_SIMPLEX, #합성할 글자 폰트
                #        fontScale=1, #합성할 글자 크기
                #        color=(0, 0, 255), #합성할 글자색
                #        thickness=2 #합성할 글자 두께
                #    )
                #
                #else:
                #    #웹캠 이미지에 글자 합성
                #    cv2.putText(
                #       image,  # 웹캠 이미지
                #        text="Close!!",  # 합성할 이미지
                #        org=(300, 100),  # 글자를 합성할 가로 세로 위치
                #        fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 합성할 글자 폰트
                #        fontScale=1,  # 합성할 글자 크기
                #       color=(0, 0, 255),  # 합성할 글자색
                #        thickness=2  # 합성할 글자 두께
                #    )


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