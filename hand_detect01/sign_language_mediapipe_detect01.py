import glob
import numpy as np
import cv2
import mediapipe as mp
import numpy.linalg as LA

#학습 데이터 저장 경로
path = "./hand_dataset/*.npy"

#glob.glob(path) : path에 저장된 파일 리스트 리턴
file_list = glob.glob(path)
print("file_list=",file_list)

#모든 학습 데이터를 저장할 리스트
all_data = []

#for file_path in file_list : file_list에서 파일 경로 1개씩 file_path에 저장
for file_path in file_list:

    print("=" *100)
    # file_path 출력
    print("file_path=", file_path)
    print("=" * 100)
    # file_path 에 저장된 파일의 내용 읽어서 data에 저장
    data = np.load(file_path)
    print("=" * 100)
    print("data=", data)
    print("=" * 100)
    # 리스트 all_data에 data 추가
    all_data.extend(data)

#np.array(all_data) : 손의 좌표와 각도가 저장된 all_data를 배열로 변환
#dtype=np.float32 : 저장된 데이터의 타입을 실수 32비트 (float32)로 변환
save_data = np.array(all_data, dtype=np.float32)
print("=" * 100)
print("save_data=",save_data)
print("=" * 100)

#save_data[: => 모든줄,    63:-1 => 63칸부터 마지막칸 -1 앞    ] 리턴 (손의 각도 데이터)
angle_arr = save_data[:, 63 :- 1]
print("=" * 100)
print("angle_arr=",angle_arr)
print("=" * 100)

#save_data[: => 모든줄,    -1 => 마지막칸 ] 리턴 (수어 종류 데이터)
label_arr = save_data[:, -1]
print("=" * 100)

print("label_arr=",label_arr)
print("=" * 100)

#예측값과 문자열
gesture = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F',
    6:'G', 7:'H', 8:'I', 9:'J'
}

#cv2.ml.KNearest_create(): KNN 객체 생성
knn = cv2.ml.KNearest_create()

#새로운 데이터와 거리를 구할 데이터 준비
#cv2.ROW_SAMPLE : 하나의 데이터가 한 행으로 구성됨
#cv2.COL_SAMPLE : 하나의 데이터가 한 열로 구성됨

#knn.train(angle_arr => 거리를 계산 할 keypoint 각도, cv2.ml.ROW_SAMPLE, Label => 각도의 손모양)
knn.train(angle_arr, cv2.ml.ROW_SAMPLE, label_arr)

#창의 크기를 수정 가능 하도록 설정 cv2.WINDOW_NORMAL
#cv2.namedWindow(원도우 창 타이틀, flags=cv2.WINDOW_NORMAL)
cv2.namedWindow(winname='webcam_window01', flags=cv2.WINDOW_NORMAL)

#창의 크기를 수정
#cv2.namedWindow(원도우 창 타이틀, width=원도우 크기 가로, height=원도우 크기 세로)
cv2.resizeWindow(winname='webcam_window01', width=1024, height=800)

#mp.solutions.hands : 화면에서 손과 손가락 관절 위치 정보를 탐지 객체 리턴
mp_hands = mp.solutions.hands

# 인식한 손의 key point 를 그릴 객체
mp_drawing = mp.solutions.drawing_utils

#cv2.VideoCapture(0) : 웹캠의 화면을 가져올 객체를 생성해서 리턴
cap = cv2.VideoCapture(0)

#with mp_hands.Hands : 여기부터 화면에서 손과 손가락 위치 탐지
with mp_hands.Hands() as hands:

    #cap.isOpened() : 웹캠이 정상적으로 동작하면 True, 웹캠이 정상적으로 동작하지 못하면 False 리턴
    while cap.isOpened()==True: #웹캠이 정상적으로 동작하는 동안 반복
        #cap.read() : 웹 캠의 화면을 가져음

        #웹캠의 화면을 정상적으로 가져오면 success에 True 가 저장
        #웹캠의 화면을 가져오는데 실패하면 success에 False가 저장

        #웹캠이 가져온 화면은 image에 저장
        success, image = cap.read()

        # cv2.flip(image, 1) : 웹캠 이미지를 좌우 반전
        image = cv2.flip(image,1) # 1은 좌우 반전, 0은 상하 반전입니다

        if success == False : #웹캠의 화면을 가져오는데 실패하면
            continue #반복문의 다음으로

        #cv2.cvtColor(image, cv2.COLOR_BGR2RGB) : 웹캠 이미지 image를 BGR에서 RGB로 변환
        #hands.process (): 웹캠에서 손과 손가락 관절 위치 탐지해서 리턴
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # results.multi_hand_landmarks : 웹 캠 화면에서 손이 없으면 None 이리턴
        # results.multi_hand_landmarks != None : 화면에 손이 존재 하면
        if results.multi_hand_landmarks != None:

            # results.multi_hand_landmarks : 탐지한 손의 keypoint 들이 저장
            # for hand_Landmarks in results.multi_hand_landmarks: 탐지한 손의 keypoint 순서대로 1개씩
            #                                                     hand_landmarks에 저장하고 반복문 실행
            for hand_landmarks in results.multi_hand_landmarks:
                # 탐지한 손의 keypoint를 화면에 그림
                mp_drawing.draw_landmarks(
                    image,  #keypoint를 그릴 이미지 (웹캠화면)
                    hand_landmarks,  #keypoint 좌표
                    mp_hands.HAND_CONNECTIONS,  #탐지한 손의 keypoint를 선으로 연결함
                    #keypoint 를 표시할 원의 모양
                    #color=(0, 255, 0) : 녹색 thickness=2 : 선두께 circle_radius=4 : 원반지름
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    #keypoint 를 표시할 원의 모양
                    #color=(255,0, 0) : 파란색 thickness=2 : 선두께 circle_radius=2 : 원반지름
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )

                #np.zeros((21, 3)) : 21줄 3칸의 0으로 초기화된 배열 생성
                joint = np.zeros((21, 3))

                # hand_Landmarks.Landmark : 손의 keypoint 좌표 리턴
                # j : keypoint °/ index
                # Lm : keypoint 좌표
                for j, lm in enumerate(hand_landmarks.landmark):
                    # j : keypoint index
                    print("j=", j)
                    # Lm: keypoint 좌표

                    # lm.x : keypoint 의 x좌표
                    print("lm.x=", lm.x)

                    # lm.y : keypoint 의 y좌표
                    print("lm.y=", lm.y)

                    # Lm.z : keypoint 의 z좌표
                    print("lm.z=", lm.z)
                    # 각도를 구하기 위해 keypoint 의 x,y,z 좌표를
                    # joint 배열 j번째 행에 대입
                    joint[j] = [lm.x, lm.y, lm.z]
                    print("=" * 100)

                print("joint=", joint)

                # v1에서 v2를 빼서 차를 계산
                # 배열의 행 인덱스                                                             #배열의 열 인덱스
                                                                                            #x,y,z
                v1 = joint[ [0, 1, 2, 3, 0, 5, 6, 7, 0, 9,  10, 11,  0, 13, 14, 15,  0, 17, 18, 19],   :]
                v2 = joint[ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],   :]
                # v2에서 v1배기
                v = v2 - v1
                print("=" * 100)
                print("v=", v)
                print("=" * 100)

                # v를 정규화 시킴
                # 정규화 결과는 1차원 배열
                v_normal = LA.norm(v, axis=1)
                print("=" * 100)
                print("v_normal=", v_normal)
                print("=" * 100)

                # v와 연산하기 위해서 v_normal을 2차원 배열로 변환
                # [:, np.newaxis]
                #                   => 모든행
                #                   np.newaxis => 차원 추가
                #모든 행의 차원을 추가해서 1차원 배열 v_normal을 2차원 배열 v_normal2로 변환
                v_normal2 = v_normal[:, np.newaxis]
                print("v_normal2=", v_normal2)

                #v를 v_normal2로 나눠서 거리를 정규화시킴
                v2 = v / v_normal2
                print("=" * 100)
                print("v2=", v2)
                print("=" * 100)

                # a,b 배열의 곱을 계산
                a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
                b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                ein = np.einsum('ij, ij->i', a, b)
                print("=" * 100)
                # 행렬의 곱 조회
                print("ein=", ein)
                print("=" * 100)

                #코사인값 계산 (라디안)
                #np.arccos(0, 5) # 1.0471975511965979 -> pi/3
                radian = np.arccos(ein)
                print("radian=",radian)

                #코사인값을 각도로 변환
                #np.degrees(1.0471975511965979) : 60eh
                angle = np.degrees(radian)
                print("angle=",angle)

                #angle을 numpy 배열로 변환
                data = np.array([angle], dtype=np.float32)
                print("data",data)

                # knn.findNearest(data, 3) : 현재 손의 각도 data 와 KNN에 저장된 angle_arr 과 거리를 계산하고
                # 가장 거리가 가까운 3개 조회
                # 거리가 가까운 손모양 결과 (실수로 라턴) retval
                # 거리가 가까운 손모양 결과 (배멸로 리턴) results
                # 거리가 가까운 3개의 손모양 결과
                # 가장 가까운 거리 3개
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

                # int(retval): 예측값을 정수로 변환
                idx = int(retval)
                print("idx=", idx)
                # image.shape[0] : 이미지 세로 사이즈
                print("image.shape[0]=", image.shape[0])
                # image.shape[1] : 이미지 가로 사이즈
                print("image. shape[1]=", image.shape[1])
                # hand_Landmarks. Landmark[0].x: 손위치 X좌표 (8~1)
                print("hand_landmarks. landmark[0].x=", hand_landmarks.landmark[0].x)
                # hand_Landmarks.Landmark[0].y: 손위차 y좌표 (8~1)
                print("hand_landmarks. landmark[0].y=", hand_landmarks.landmark[0].y)

                # 예측값이 0~9 사이이면
                if 0 <= idx <= 9:
                    cv2.putText(image,
                                text=gesture[idx],  # 예즉값
                                org=(
                                    int(hand_landmarks.landmark[0].x * image.shape[1]),  # 손위치 x좌표
                                    int(hand_landmarks.landmark[0].y * image.shape[0])  # 손위치 y좌표
                                ),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 글자체
                                fontScale=1,  # 글자크기
                                color=(0, 0, 255),  # 글자색
                                thickness=2  # 폰트 두까
                                )
                    
        # cv2.imshow (원도우창 타이틀, 원도우창에 출력할 이미지) : 웹캠 화면을 화면에 출력
        cv2.imshow('webcam_window01', image)


        #cv2.waitKey(1) : 사용자가 키보드 입력하도록 1초 기다림
        #                 기다리는 시간동안 사용자가 키보드 입력을 하면 입력한 키보드값을 리턴
        #                 기다리는 시간동안 사용자가 입력한 키보드가 없으면 None 리턴

        if cv2.waitKey(1) == ord('q'):  #키보드 입력이 q이면
            break  # 반복문 종료

#웹캠의 화면 그만 가져오도록 설정
cap.release()