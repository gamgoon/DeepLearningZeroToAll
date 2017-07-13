import tensorflow as tf

# X and Y data (학습데이터)
x_train = [1,2,3]
y_train = [1,2,3]

# 가중치와 편향 변수 tf.Variable 에는 shape 을 지정
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW + b
hypothesis = x_train * W + b

# cost/loss function, reduce_mean : 평균계산
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize cost 함수의 값을 최소화하기 위한 가중치와 편향의 값 조절
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
