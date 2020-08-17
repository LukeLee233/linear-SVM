import tensorflow as tf
import numpy as np

X_train = np.linspace(-1, 1, 101)
y_train = np.asmatrix(8 * X_train + np.random.randn() * 2 + 139).T
X_train = np.asmatrix(X_train).T

n_dim = X_train.shape[1]

with tf.name_scope("leaning-rate"):
    lr = tf.constant(0.01, dtype=tf.float32)

with tf.name_scope("Inputs"):
    X = tf.placeholder(tf.float32, [X_train.shape[0], n_dim])
    y = tf.placeholder(tf.float32, [X_train.shape[0], 1])

with tf.name_scope("linear-layer"):
    w = tf.Variable(np.ones([n_dim, 1]), dtype=tf.float32)
    b = tf.Variable(0, dtype=tf.float32)
    yhat = tf.matmul(X, w) + b

with tf.name_scope("MSE-loss"):
    loss = tf.reduce_mean(tf.square(yhat - y))
    tf.summary.scalar('loss', loss)

with tf.name_scope("gradient-optimizer"):
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

init = tf.initialize_all_variables()
merged = tf.summary.merge_all()

# initialize session
with tf.Session() as sess:
    with tf.summary.FileWriter("./logs", graph=sess.graph) as writer:
        num_epochs = 500
        sess.run(init)

        for epoch in range(num_epochs):
            _, statics = sess.run([train_op, merged], feed_dict={X: X_train, y: y_train})
            writer.add_summary(statics, epoch)

        print("W: %.4f" % w.eval(sess))
        print("b: %.4f" % b.eval(sess))
        print("MSE: %.4f" % loss.eval(feed_dict={X: X_train, y: y_train}, session=sess))
