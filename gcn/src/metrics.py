import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import sys


def masked_softmax_cross_entropy(preds, labels, mask=None, model=None):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    if mask is not None:
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(y_pred, y_true):
    ###change float prediction score to binary(0 and 1)###
    numlabel = np.sum(y_true, axis=1) # number of labels of each instance
    binary_pred = np.zeros(y_pred.shape, dtype=np.int)
    for index in range(y_pred.shape[0]):
        instance_temp = np.copy(y_pred[index])
        num_label_temp = int(numlabel[index])
        for label in range(num_label_temp):
            max_index = np.argmax(instance_temp)
            instance_temp[max_index] = -sys.maxsize
            binary_pred[index][max_index] = 1
    # for index in range(y_pred.shape[0]):
    #     if not (binary_pred[index]==y_true[index]).all():
    #         print('@@@', index)
    #         print('binary_pred[index]', binary_pred[index])
    #         print('y_true[index]', y_true[index])
    #         exit(1)
    f1_micro = f1_score(y_true, binary_pred, average='micro')
    f1_macro = f1_score(y_true, binary_pred, average='macro')
    return f1_micro, f1_macro


def print_mat(model, mat):
    # zero = tf.constant(0, dtype=tf.float32)
    # where = tf.not_equal(mat, zero)
    # result = tf.boolean_mask(mat, where)

    result = mat[0:100,0:100]

    model.printer = tf.Print(result, [result],
                             message='result:\n',
                             summarize=100*100)