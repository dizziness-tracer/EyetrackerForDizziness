import cv2
import pandas as pd
import numpy as np

video_path = './동영상/1차/1_n1.mp4'
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

#setting ROI ROi를 설정해서 rect로 변환

x = int(img.shape[0] / 10)
y = int(img.shape[1] / 10)
print(img.shape[0],img.shape[1])
print(x,y)

out = cv2.selectROI('Window',img,fromCenter=True,showCrosshair=True)
img = img[out[1]:(out[3]+out[1]),out[0]:(out[0]+out[2])]
# img = img[(out[1]-y):(out[1]+y),(out[0]-x):(out[0]+x)]
img = cv2.resize(img,dsize=(680,240),interpolation=cv2.INTER_AREA)
cir = cv2.selectROI('Window',img, fromCenter=True, showCrosshair=True)
print(cir)
cv2.destroyWindow('Window')
# tracker 초기화
tracker.init(img, cir)

list_left = []
list_top = []

list_x=[]
list_y=[]
result_X = []
result_y = []
i = -1
j = -1
count = 0
while True:
    ret, img = cap.read()
    if not ret:
        exit()
    img = img[out[1]:(out[3] + out[1]), out[0]:(out[0] + out[2])]
    img = cv2.resize(img, dsize=(680, 240), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    success, box = tracker.update(img) # success 는 boolean , box는 데이터 값
    left, top, w, h =[int(v) for v in box]
    # cv2.circle(img,(int((left+w/2)),int((top+h/2))),int(w/2),(255,255,0),2)
    x = left + w / 2
    if x < 300 or x > 370: #중앙 x값을 300으로 지정
        cv2.imwrite("./1_n1 %d.jpg" % count, img)
#     cv2.imwrite("./frame/ab/ap-1.R %d.jpg" % count, img)
    count += 1
    cv2.rectangle(img, pt1=(left, top),pt2 =(left+w, top+h),color = (255,255,0),thickness=2)
    print(left+w / 2)
    # print("(x,y):",left+w/2,top+h/2) #중심의 좌표값


    list_left.append(left)
    list_top.append(top)
    list_x.append(left + w / 2)
    list_y.append(top + h / 2)
    cv2.imshow('window', img)
    # print("(x,y):",left+w/2,top+h/2) #중심의 좌표값
    # i = i + 1
    # j = j + 1
    # if i > 0 and i < len(list_x):
    #     result_X.append(list_x[i] - list_x[i - 1])
    # if j > 0 and j < len(list_y):
    #     result_y.append(list_y[i] - list_y[i - 1])

    data = {"left좌표":list_left, "top좌표":list_top, "x좌표": list_x, "y_좌표": list_y}
    df = pd.DataFrame(data)
    # df_min_max = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    # df.to_csv('이석증4.2.csv', header=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break#그냥 꺼져 버릴 수 있음
cap.release()
cv2.destroyWindow()