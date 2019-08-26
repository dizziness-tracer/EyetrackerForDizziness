import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.core import Activation, Dense
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping

training_data = pd.read_csv('train.csv')
target_data = pd.read_csv('test.csv')
print(type(training_data))
print(np.shape(training_data))
print(np.shape(target_data))

training_data = training_data.values
target_data = target_data.values

print(training_data)
print(target_data)

X_train, y_train = training_data[:, 0:-1], training_data[:, [-1]]
X_test, y_test = target_data[:, 0:-1], target_data[:, [-1]]

print(y_train)
print(np.shape(X_train))

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

model = Sequential()
model.add(Dense(512, input_shape=(2,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, batch_size=50, validation_data=(X_test, y_test),
          callbacks=[early_stopping_callback, checkpointer])

model.summary()

print(model.predict(X_test))

results = model.evaluate(X_test, y_test)

print('loss: ', results[0])
print('accuracy: ', results[1])