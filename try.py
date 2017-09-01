# #!/usr/bin/env python3.6

# import tensorflow as tf
# import numpy as np
# from sklearn.preprocessing import normalize

# '''
# Exp with mat mul for convolution.
# '''

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
#     whole = np.identity(10)
#     print('whole\n', whole)
#     print('portion\n', whole[0:4,0:4])
#     print('sklearn normalized\n', normalize(np.array([[1, 0], [1, 1]]), norm='l1'))

# # matrix1 = np.array([[[0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], \
# #     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
# #     [0, 0, 0, 0, 0, 0]], \
# #     [[0, 2, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0], \
# #     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
# #     [0, 0, 0, 0, 0, 0]]])
# # matrix2 = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).reshape((2, 6, 1))
# # print(matrix1*matrix2)


import numpy as np
import tensorflow as tf
input = np.array([[1,0,3,5,0,8,6], [0,0,1,1,1,1,1]])
X = tf.placeholder(tf.float32,[2,7])
zeros = tf.cast(tf.zeros_like(X),dtype=tf.bool)
ones = tf.cast(tf.ones_like(X),dtype=tf.bool)
loc = tf.where(input!=0,ones,zeros)
result=tf.boolean_mask(input,loc)
with tf.Session() as sess:
 out = sess.run([loc],feed_dict={X:input})
 print (np.array(out))
