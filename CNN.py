import os
import cv2
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model

x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")


model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', input_shape=x_train.shape[1:]))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=100, nb_epoch=2, validation_data = (x_test, y_test))
model.summary()

predict = model.predict_classes(x_test)

#for i in range(len(test)):
#    print(name[i] + ' predict : ' + str(classes[predict[i]]))

results = model.evaluate(x_test, y_test)

print('loss: ', results[0])
print('accuracy: ', results[1])


import matplotlib.font_manager as fm

fontprop = fm.FontProperties(fname="IropkeBatangM.ttf", size=13)

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='loss') #train loss
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(hist.history['val_acc'], 'b', label='test_accuracy') #train acc
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc= 'lower left')

plt.title('CNN_test - 정확도 & 손실도', fontproperties=fontprop)
plt.show()