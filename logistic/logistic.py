import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X_train = np.linspace(-1, 1, 10)
y_train = np.asmatrix([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).T
X_train = np.asmatrix(X_train).T

n_dim = X_train.shape[1]

with tf.name_scope("leaning-rate"):
    lr = tf.constant(0.01, dtype=tf.float32)

with tf.name_scope("Inputs"):
    X = tf.placeholder(tf.float32, [X_train.shape[0], n_dim])
    y = tf.placeholder(tf.float32, [X_train.shape[0], 1])

with tf.name_scope("logistic-layer"):
    w = tf.Variable(np.ones([n_dim, 1]), dtype=tf.float32)
    b = tf.Variable(0, dtype=tf.float32)
    yhat = (1. / (1 + tf.exp(tf.matmul(X, w) + b)))

with tf.name_scope("MSE-loss"):
    loss = tf.reduce_mean(tf.square(yhat - y))
    tf.summary.scalar('loss', loss)

with tf.name_scope("gradient-optimizer"):
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)


# initialize session
init = tf.initialize_all_variables()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    with tf.summary.FileWriter("./logs",graph=sess.graph) as writer:
        num_epochs = 3000
        sess.run(init)

        for epoch in range(num_epochs):
            _, statics = sess.run([train_op, merged], feed_dict={X: X_train, y: y_train})
            writer.add_summary(statics, epoch)

        w = sess.run(w)
        b = sess.run(b)
        print("W: %.4f" % w)
        print("b: %.4f" % b)

        print(1. / (1 + np.exp(np.dot(X_train, w) + b)))
