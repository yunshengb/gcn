#!/usr/bin/env python3.6

# import tensorflow as tf
# import numpy as np
# from sklearn.preprocessing import normalize


# matrix1 = tf.constant([[[[0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0]], \
#     [[0, 2, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0]]], \
#     # Another filter/feature
#     [[[0, 3, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0]], \
#     [[0, 4, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0]]]])


# # Create another Constant that produces a 2x1 matrix.
# # matrix2 = tf.constant([[[1], [2], [3], [4], [5], [6]], \
#     # [[7], [8], [9], [10], [11], [12]]])

# # matrix2 = tf.constant([1, 2, 3, 4, 5, 6])
# matrix2 = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
# matrix2 = tf.reshape(matrix2, (2, 1, 6))

# # Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# # The returned value, 'product', represents the result of the matrix
# # multiplication.
# # product = tf.matmul(matrix1, matrix2)
# # product = (matrix1*matrix2)
# product = tf.multiply(matrix1, matrix2)

# reduced = tf.reduce_sum(product, [1, 3])
# reduced = tf.transpose(reduced)

# reduced = tf.cast(reduced, tf.float32)
# normalized = tf.nn.l2_normalize(reduced, dim=1)

# argmax = tf.argmax(reduced, 1)

# softmax = tf.nn.softmax(normalized)

# truth = tf.constant([[0.5,0.5],[1.0,0],[0.7,0.3],[0,0]])

# logits = tf.constant([[1.0,1.0],[0,-999999999999999999],[-100,100],[0,0]])

# new_softmax = tf.nn.softmax(logits)

# loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=truth)

# reduce_mean = tf.reduce_mean(loss)

# with tf.Session() as sess:
#     print(matrix1.shape, matrix2.shape)
#     result = sess.run(product)
#     print('product\n', result)
#     result = sess.run(reduced)
#     print('reduced\n', result)
#     result = sess.run(normalized)
#     print('normalized\n', result)
#     result = sess.run(argmax)
#     print('argmax\n', result)
#     result = sess.run(softmax)
#     print('softmax\n', result)
#     result = sess.run(logits)
#     print('logits\n', result)
#     result = sess.run(new_softmax)
#     print('new_softmax\n', result)
#     result = sess.run(truth)
#     print('truth\n', result)
#     result = sess.run(loss)
#     print('loss\n', result)
#     result = sess.run(reduce_mean)
#     print('reduce_mean\n', result)
# whole = np.identity(10)
# print('whole\n', whole)
# print('portion\n', whole[0:4,0:4])
# print('sklearn normalized\n', normalize(np.array([[1, 0], [1, 1]]), norm='l1'))

# matrix1 = np.array([[[0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0]], \
#     [[0, 2, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0]]])
# matrix2 = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).reshape((2, 6, 1))
# print(matrix1*matrix2)


# import numpy as np
# import tensorflow as tf
# input = np.array([[1,0,3,5,0,8,6], [0,0,1,1,1,1,1]])
# X = tf.placeholder(tf.float32,[2,7])
# zeros = tf.cast(tf.zeros_like(X),dtype=tf.bool)
# ones = tf.cast(tf.ones_like(X),dtype=tf.bool)
# loc = tf.where(input!=0,ones,zeros)
# result=tf.boolean_mask(input,loc)
# with tf.Session() as sess:
#  out = sess.run([loc],feed_dict={X:input})
#  print (np.array(out))

import numpy as np
from sklearn.preprocessing import normalize


def proc_adj(adj, weights=[0.7, 0.2, 0.1]):
    one = adj
    self = np.identity(adj.shape[0])
    one_with_self = adj + self
    temp = one_with_self.dot(one_with_self)
    temp = div0(temp, temp)
    two = temp - one_with_self
    d = one.sum(1)
    normalized_adj = np.zeros(adj.shape)
    for i, neighbor in enumerate([self, one, two]):
        normalized_adj += norm(neighbor, d, weights[i])
    return normalized_adj

def norm(neighbor, d, weight):
    return weight * normalize(np.multiply(neighbor, d), norm='l1')

def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

adj = np.array(
    [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [0, 1, 1, 0, 1],
     [0, 0, 0, 1, 0]])
adj_normalized = proc_adj(adj)
