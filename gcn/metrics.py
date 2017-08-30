import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask, model=None):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    if model:
        probs = tf.nn.softmax(preds)
        model.printer = tf.Print(probs, [probs[0:100,0:100]],
                                 message='probs:\n',
                                 summarize=100*100)
        model.probs = probs
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

    #
    # if __name__ == '__main__':
    #
    #
