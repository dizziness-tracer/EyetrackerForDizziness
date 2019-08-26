import os
import numpy as np
import pandas as pd
import cv2

folder_path = './train_frame/'
classes = ['ab', 'n']
num_classes = len(classes)

test_folder_path = './test_frame/'
test_classes = ['ab', 'n']
test_num_classes = len(test_classes)

image_w = 68
image_h = 24
test_image_w = 68
test_image_h = 24

x_train = []  # 정상, 비정상의 픽셀데이터값
y_train = []  # label
x_test = []
y_test = []

for index, cla in enumerate(classes):
    label = [0 for i in range(num_classes)]
    label[index] = 1
    image_dir = folder_path + cla + '/'
    for top, dir, f in os.walk(image_dir):  # os를 사용해서 사진을 리스트로 바꾼다
        for filename in f:
            image_dir + filename
            img = cv2.imread(image_dir + filename)
            img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
            x_train.append(img / 256)
            y_train.append(label)

for test_index, test_cla in enumerate(test_classes):
    test_label = [0 for i in range(test_num_classes)]
    test_label[test_index] = 1
    test_image_dir = test_folder_path + test_cla + '/'
    for top, dir, f in os.walk(test_image_dir):  # os를 사용해서 사진을 리스트로 바꾼다
        for test_filename in f:
            test_image_dir + test_filename
            test_img = cv2.imread(test_image_dir + test_filename)
            test_img = cv2.resize(test_img, None, fx=test_image_w / test_img.shape[1],
                                  fy=test_image_h / test_img.shape[0])
            x_test.append(test_img / 256)
            y_test.append(test_label)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

np.save('x_train', x_train)
np.save('y_train', y_train)
np.save('x_test', x_test)
np.save('y_test', y_test)