import numpy as np
import pandas as pd
import cv2

video_path = '환자2-train.mp4'
cap = cv2.VideoCapture(video_path)

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

if not cap.isOpened():
    exit()

tracker = cv2.TrackerCSRT_create()

ret, img = cap.read()
cv2.namedWindow('Window')
cv2.imshow('Window', img)

# setting ROI ROi를 설정해서 rect로 변환

cir = cv2.selectROI('Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Window')

# tracker 초기화
tracker.init(img, cir)

result_x = []
result_y = []
list_x = []
list_y = []
i = -1
j = -1

while True:
    ret, img = cap.read()

    if not ret:
        exit()
    success, box = tracker.update(img)  # 따라가게 하기(tracker)
    # success 는 boolean(성공유무) , box는 데이터 값
    left, top, w, h = [int(v) for v in box]  # 왼,위,너비,높이
    #     cv2.rectangle(img, pt1=(left,top),pt2 =(left+w, top+h),color = (255,255,0),thickness=2)
    cv2.circle(img, (int((left + w / 2)), int((top + h / 2))), int(w / 2), (255, 255, 0), 0)
    # pt1은 왼쪽아래,pt2는 오른쪽위  좌표

    # print("(x,y):",left+w/2,top+h/2) #중심의 좌표값
    list_x.append(left + w / 2)
    i = i + 1
    list_y.append(top + h / 2)
    j = j + 1
    if i > 0 and i < len(list_x):
        result_x.append(list_x[i] - list_x[i - 1])
    if j > 0 and j < len(list_y):
        result_y.append(list_y[i] - list_y[i - 1])
    cv2.imshow('img', img)
    data = {"x좌표": result_x, "y_좌표": result_y}
    df = pd.DataFrame(data)
    df.to_csv('환자2.1-train.csv', header=None)

    if cv2.waitKey(1) == ord('q'):
        break  # q누르면 종료