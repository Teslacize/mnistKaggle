import numpy as np
import csv
import tensorflow as tf
from tensorflow.python import debug as tf_debug

learning_rate = 0.000001
epochs = 10
batch_size = 100
X = []
with open('train.csv') as f:
	reader = csv.reader(f)
	reader.__next__()
	for row in reader:
		X.append([int(i) for i in row])
	

my_mat = X
X = np.asarray(my_mat)
y_train = np.ndarray(shape=(X.shape[0],10))
y_train.fill(0)
for i in range(X.shape[0]):
	y_train[i][X[i,0]] = 1
X_train = X[:,1:]
y_train = np.asarray(y_train)

X = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

dnn1 = tf.layers.dense(inputs=X, units=100, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=dnn1, units=10)

y_ = tf.nn.softmax(logits)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_) + (1 - y)*tf.log(1 - y_), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
init_op = tf.global_variables_initializer()
sess.run(init_op)
test = tf.to_int32(tf.shape(y)[0])
total_batch = int(X_train.shape[0]/batch_size)
print(total_batch)
print('Training\n')
for epoch in range(epochs):
	avg_cost = 0
	for i in range(total_batch):
		batch_x = X_train[i*batch_size:(i+1)*batch_size,:]
		batch_y = y_train[i*batch_size:(i+1)*batch_size,:]
		print
		_, c = sess.run([optimizer, cross_entropy], feed_dict={X:batch_x, y:batch_y})
		# print(c)
		avg_cost += c/total_batch
	print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
print(sess.run(accuracy, feed_dict={X: X_train, y: y_train}))

sess.close()







