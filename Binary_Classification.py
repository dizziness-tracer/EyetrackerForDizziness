import tensorflow as tf
import numpy as np
import os

tf.set_random_seed(777)  # for reproducibility

# 학습 데이터 로딩
xy_train = np.loadtxt('train_clean.csv', delimiter=',', dtype=np.float32)
xy_test = np.loadtxt('test_clean.csv', delimiter=',', dtype=np.float32)

x_train = xy_train[:, 0:-1]
y_train = xy_train[:, [-1]]

x_test = xy_test[:, 0:-1]
y_test = xy_test[:, [-1]]

# 플레이스홀더로 그래프 설정
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 학습 파라미터 초기화
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설 함수 그래프 설정
# sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# 비용 함수 그래프 설정
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# 최소값 구하는 함수 그래프 설정
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 예측값과 정확도 그래프 설정
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# 그래프 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("-------------------------------------------")
print(" Loss values tracing ")
print("-------------------------------------------")
for step in range(10001):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
    if step % 200 == 0:
        print(step, cost_val)

print("-------------------------------------------")

print(tf.concat([hypothesis, predicted], 0))

# 학습 후 최종 결과 값 출력
Rst_hypothesis, Rst_predict, Rst_accuracy = sess.run([hypothesis, predicted, accuracy],
                                                     feed_dict={X: x_test, Y: y_test})
print("\nHypothesis: ", Rst_hypothesis, "\nCorrect (Y): ", Rst_predict, "\nAccuracy: ", Rst_accuracy)
