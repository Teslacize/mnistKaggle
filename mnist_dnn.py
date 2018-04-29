import numpy as np
import csv
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.1
epochs = 10
batch_size = 100
# X = []
# with open('train.csv') as f:
# 	reader = csv.reader(f)
# 	reader.__next__()
# 	for row in reader:
# 		X.append([int(i) for i in row])
	

# my_mat = X
# X = np.asarray(my_mat)
# y_train = np.ndarray(shape=(X.shape[0],10))
# y_train.fill(0)
# for i in range(X.shape[0]):
# 	y_train[i][X[i,0]] = 1
# X_train = X[:,1:]
# y_train = np.asarray(y_train)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Read in complete!")


X = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784,100], stddev = 0.5), name = "W1")
b1 = tf.Variable(tf.random_normal([100], stddev = 0.5), name = "b1")

hidden1 = tf.add(tf.matmul(X,W1),b1)
H1 = tf.nn.relu(hidden1, name="H1")

W2 = tf.Variable(tf.random_normal([100,10], stddev = 0.5), name = "W2")
b2 = tf.Variable(tf.random_normal([10], stddev = 0.5), name = "b2")

hidden2 = tf.add(tf.matmul(H1,W2),b2)
y_ = tf.nn.softmax(hidden2, name="H2")
y_ = tf.clip_by_value(y_, 1e-10, 0.99999)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_) + (1 - y)*tf.log(1 - y_), axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
init = tf.global_variables_initializer()

sess.run(init)
total_batch = int(len(mnist.train.labels)/batch_size)
for epoch in range(epochs):
	avg_cost = 0
	for i in range(total_batch):
		X_batch, y_batch = mnist.train.next_batch(batch_size=i)
		sess.run([optimizer,cross_entropy], feed_dict={X:X_batch, y:y_batch})
		avg_cost += c/total_batch
		print(c)
	print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
sess.close()







