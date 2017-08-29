#!/usr/bin/env python3.6

import tensorflow as tf
import numpy as np

'''
Exp with mat mul for convolution.
'''

matrix1 = tf.constant([[[[0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0]], \
    [[0, 2, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0]]], \
    # Another filter/feature
    [[[0, 3, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0]], \
    [[0, 4, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0]]]])


# Create another Constant that produces a 2x1 matrix.
# matrix2 = tf.constant([[[1], [2], [3], [4], [5], [6]], \
    # [[7], [8], [9], [10], [11], [12]]])

# matrix2 = tf.constant([1, 2, 3, 4, 5, 6])
matrix2 = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
matrix2 = tf.reshape(matrix2, (2, 1, 6))

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
# product = tf.matmul(matrix1, matrix2)
# product = (matrix1*matrix2)
product = tf.multiply(matrix1, matrix2)

reduced = tf.reduce_sum(product, [1, 3])
reduced = tf.transpose(reduced)

with tf.Session() as sess:
    print(matrix1.shape, matrix2.shape)
    result = sess.run(product)
    print(result)
    result = sess.run(reduced)
    print(result)

# matrix1 = np.array([[[0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0]], \
#     [[0, 2, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
#     [0, 0, 0, 0, 0, 0]]])
# matrix2 = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).reshape((2, 6, 1))
# print(matrix1*matrix2)

