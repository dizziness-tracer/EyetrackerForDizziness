import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.core import Activation, Dense, Dropout

training_data = pd.read_csv('train.csv')
target_data = pd.read_csv('test.csv')
print(type(training_data))
print(np.shape(training_data))
print(np.shape(target_data))

training_data= training_data.values
target_data = target_data.values

print(training_data)
print(target_data)

X_train, y_train = training_data[:,0:-1], training_data[:,[-1]]
X_test, y_test = target_data[:,0:-1], target_data[:,[-1]]

print(y_train)
print(np.shape(X_train))


model = Sequential()
model.add(Dense(128, input_shape =(2,), activation='relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.05))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.02))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs = 30, batch_size = 500, validation_data = (X_test, y_test))

print('\nAccuracy: {:.4f}'.format(model.evaluate(X_test, y_test)[1]))
a = model.predict(X_test)
print(a)

results = model.evaluate(X_test, y_test)

print('loss: ', results[0])
print('accuracy: ', results[1])

