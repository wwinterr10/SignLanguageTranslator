from tkinter import *
import cv2
import mediapipe as mp

import numpy as np
import os
import numpy.linalg as LA

#os.makedirs('hand_dataset', exist_ok=True) : hand_dataset 디렉토리 생성
#exist_ok=True : hand_dataset 디렉토리가 이미 존재한다면 디렉토리 만들지않음
os.makedirs('hand_dataset', exist_ok=True)

#현재 프레임을 저장할 변수
frame = 0

#600 프레임 데이터 생성할 것임
MAX_FRAME = 600

#media pipe 데이터를 저장할 리스트
all_data = []

#입력한 문자를 저장할 전역 변수
action = "미정"

#버튼 클릭시 실행되는 함수 구현
def btnpress():


    #전역변수에 값을 대입하기 위해서 선언
    global action

    print("버튼 클릭했음!!!")

    #ent.get() : ent에 입력한 값 리턴
    input = ent.get()
    print("input=",input)

    #전역변수 action에 input 저장
    action = input

    #윈도우창 종료
    window.destroy()

#윈도우 창 생성
window = Tk()

#Entry(window) : 입력 박스 생성
ent = Entry(window)
#window 창에 입력 박스 추가
ent.pack()

#Label(window) : 레이블 생성
label = Label(window)
#레이블의 문자열 설정
label.config(text="데이터 입력할 알파벳을 입력하세요")
#window 창에 레이블 추가
label.pack()

btn = Button(window)
#버튼 텍스트 생성
btn.config(text="확인")

#버튼 클릭시 btnpress 함수 실행
btn.config(command=btnpress)
#window 창에 버튼 추가
btn.pack()

#window 창 실행
window.mainloop()

#전역변수 action 출력
print("action=",action)

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

        #프레임 1 증가
        frame = frame + 1

        #action별로 600프레임씩 데이터 저장
        #MAX_FRAME은 600
        if frame >= MAX_FRAME:
            break

        # cap.read() : 웹캠 화면을 가져옴

        # 웹캠의 화면을 정상적으로 가져오면 success에 True가 저장
        # 웹캠의 화면을 가져오는데 실패하면 success에 False가 저장

        # 웹캠의 가져온 화면을 image에 저장
        success, image = cap.read()

        # cv2.flip(image, 1) :웹캠 이미지를 좌우 반전
        image = cv2.flip(image, 1)  # 1은 좌우 반전, 0은 상하 반전입니다

        if success == False:  # 웹캠의 화면을 가져오는데 실패하면
            continue  # 반복문의 다음으로

        # cv2.cvtColor(image, cv2.COLOR_BGR2RGB) : 웹캠 이미지 image를 BGR에서 RGB로 변환
        # hands.process () : 웹캠에서 손과 손가락 관절 위치 탐지해서 리턴
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


        if results.multi_hand_landmarks != None:
            #print("hello")
            cv2.putText(
                image,  # 웹캠 이미지
                text="Detect Hand!!",  # 합성할 글자
                org=(300, 50),  # 글자를 합성할 가로 세로 위치
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 합성할 글자 폰트
                fontScale=1,  # 합성할 글자 크기
                color=(0, 0, 255),  # 합성할 글자색
                thickness=2  # 합성할 글자 두께
            )
            cv2.putText(
                image,  # 웹캠 이미지
                text=f"Gathering {action} Data Frame: {MAX_FRAME - frame} Left",  # 합성할 글자
                org=(0, 100),  # 글자를 합성할 가로 세로 위치
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 합성할 글자 폰트
                fontScale=1,  # 합성할 글자 크기
                color=(0, 0, 255),  # 합성할 글자색
                thickness=2  # 합성할 글자 두께
            )

            # results.multi_hand_landmarks : 탐지한 손의 keypoint들이 저장
            # for hand_landmarks in results.multi_hand_landmarks: 탐지한 손의 keypoint 순서대로 1개씩 hand_landmarks에 저장하고 반복문 실행

            for hand_landmarks in results.multi_hand_landmarks:
                # 탐지한 손의 keypoint를 화면에 그림
                mp_drawing.draw_landmarks(
                    image,  # keypoint를 그릴 이미지 (웹캠화면)
                    hand_landmarks,  # keypoint 좌표
                    mp_hands.HAND_CONNECTIONS,  # 탐지한 손의 keypoint를 선으로 연결함
                    # keypoint를 표시할 원의 모양
                    # color=(0, 255, 0) : 녹색 thickness=2 : 선두께 circle_radius=2 : 원반지름
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                    # keypoint를 표시할 원의 모양
                    # color=(255, 0, 0) : 파란색 thickness=2 : 선두께 circle_radius=2 : 원반지름
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )

                # np.zeros((21, 3)) : 21줄 3칸의 0으로 초기화된 배열 생성
                joint = np.zeros((21, 3))

                # hand_landmarks.landmark : 손의 keypoint 좌표 리턴
                # j : keypoint의 index
                # lm : keypoint 좌표
                for j, lm in enumerate(hand_landmarks.landmark):
                    # j : keypoint index
                    print("j=", j)
                    # lm : keypoint 좌표
                    print("lm=", lm)
                    # lm.x : keypoint의 x좌표
                    print("lm.x=", lm.x)

                    # lm.y : keypoint의 y좌표
                    print("lm.y=", lm.y)

                    # lm.z : keypoint의 z좌표
                    print("lm.z=", lm.z)
                    # 각도를 구하기 위해 keypoint의 x,y,z 좌표를
                    # joint배열 j번째 행에 대입
                    joint[j] = [lm.x, lm.y, lm.z]
                    print("=" * 100)

                print("joint=",joint)

                # v1에서 v2를 빼서 차를 계산
                # 배열의 행 인덱스                  #배열의 열 인덱스
                # x,y,z
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                # v2에서 v1빼기
                v = v2 - v1
                # print("=" *100)
                # print("v", v)
                # print("=" *100)

                # v를 정규화시킴
                # 정규화 결과는 1차원 배열
                v_normal = LA.norm(v, axis=1)
                print("=" * 100)
                print("v_normal=", v_normal)
                print("=" * 100)

                # v와 연산하기 위해서 v_normal을 2차원 배열로 변환
                # [:, np.newaxis]
                #                  : => 모든 행
                #                  np.newaxis => 차원 추가
                # 모든 행의 차원을 추가해서 1차원 배열 v_normal을 2차원 배열 v_normal2로 변환
                v_normal2 = v_normal[:, np.newaxis]
                print("v_normal2=", v_normal2)

                # v를 v_normal2로 나눠서 거리를 정규화시킴
                v2 = v / v_normal2
                print("=" * 100)
                print("v2", v2)
                print("=" * 100)

                # a,b 배열의 곱을 계산
                a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
                b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                ein = np.einsum('ij,ij->i', a, b)
                # print("=" *100)
                # 행렬의 곱 조회
                # print("ein=", ein)
                # print("=" *100)

                # 코사인값 계산 (라디안)
                # np.arccos(0,5) #1.0417975511965979 -> pi/3
                radian = np.arccos(ein)
                print("radian=", radian)

                # 코사인값을 각도로 변환
                # np.degrees(1,0471975511965979) : 60도
                angle = np.degrees(radian)
                print("angle=", angle)

                # ord(action) :action을 숫자로 변환
                action_num = ord(action)
                print("=" * 100)
                print("action_num=", action_num)
                print("=" * 100)

                # action_num에서 대문자 A를 숫자로 변환한 65를 빼줌
                action_label = action_num - ord('A')
                print("=" * 100)
                print("action_label=", action_label)
                print("=" * 100)

                # np.append(angle, index) 관절의 각도 angle과 action_label를 합침
                angle_label = np.append(angle, action_label)
                print("=" * 100)
                print("action_label=", action_label)
                print("=" * 100)

                # joint.flatten() : 관점 좌표를 1차원 배열로 변환
                # np.concatenate([joint.flatten(), angle_label]) : 관절 좌표와 각도를 data에 저장
                data = np.concatenate([joint.flatten(), angle_label])

                print("=" * 100)
                print("joint=", joint)
                print("=" * 100)
                print("joint.flatten()=", joint.flatten())
                print("=" * 100)
                print("data=", data)
                print("=" * 100)

                # 관절 좌표와 각도를 all_data에 추가
                all_data.append(data)

        cv2.imshow("webcam_window01", image)

        if cv2.waitKey(1) == ord('q'):
            break

import time
#time.time() : 현재 날짜와 시간 리턴
created_time = int(time.time())


#손의 좌표와 각도가 저장된 all_data를 hand_dataset 디렉토리에
#action 명으로 저장
np.save(os.path.join('./hand_dataset', f'{action}_{created_time}'), all_data)