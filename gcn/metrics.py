import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask, model=None):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    x = tf.nn.sampled_softmax_loss
    if model:
        model.preds = preds
        probs = tf.nn.softmax(preds)
        model.probs = probs
        # probs = tf.multiply(probs, labels)
        print_mat(model, probs)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def print_mat(model, mat):
    # zero = tf.constant(0, dtype=tf.float32)
    # where = tf.not_equal(mat, zero)
    # result = tf.boolean_mask(mat, where)

    result = mat[0:100,0:100]

    model.printer = tf.Print(result, [result],
                             message='result:\n',
                             summarize=100*100)