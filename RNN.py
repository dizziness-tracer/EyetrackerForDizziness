import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import SimpleRNN, GlobalMaxPool1D, Dropout, Flatten
import tensorflow as tf

# convert into dataset matrix
def convertToMatrix(data, step):
    X, y =[], []
    for i in range(len(data)-step):
        d=i+step
        X.append(data[i:d,])
        y.append(data[d,])
    return np.array(X), np.array(y)

step = 2

training_data = pd.read_csv('train_clean.csv')
target_data = pd.read_csv('test_clean.csv')

training_data= training_data.values
target_data = target_data.values

X_train, y_train = training_data[:,0:-1], training_data[:,[-1]]
X_test, y_test = target_data[:,0:-1], target_data[:,[-1]]

# add step elements into train and test
target_data = np.append(target_data,np.repeat(target_data[-1,],step))
training_data= np.append(training_data,np.repeat(training_data[-1,],step))

X_train, y_train = convertToMatrix(training_data,step)
X_test, y_test =convertToMatrix(target_data,step)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


model = Sequential()
model.add(SimpleRNN(units=128, input_shape=(1,step), activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

#모델 학습
optimizer_history = model.fit(X_train, y_train, epochs=300, batch_size=1500,verbose=1, validation_data = (X_test, y_test))

#모델 평가
results = model.evaluate(X_test, y_test)

print('loss: ', results[0])
print('accuracy: ', results[1])



# model.fit(X_train,y_train, epochs=200, batch_size=16, verbose=2)
# trainPredict = model.predict(X_train)
# testPredict= model.predict(X_test)
# predicted=np.concatenate((trainPredict,testPredict),axis=0)

# trainScore = model.evaluate(X_test, y_test, verbose=0)
# print(trainScore)

# model = tf.keras.Sequential()
# # model.add(Input(shape=(2,)))
# # model.add(Embedding(10000, 32))
# model.add(SimpleRNN(units=32, input_shape=(2,), return_sequences=True))
# model.add(Dense(2, activation= 'sigmoid'))


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.fit(X_train, y_train, epochs = 20, batch_size = 20, validation_data = (X_test, y_test))


# model.summary()

# print(model.predict(X_test))