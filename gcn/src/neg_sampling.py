import math
import tensorflow as tf
import numpy as np
from random import shuffle

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model:
    def __init__(self):
        pass


def test():
    model = Model()
    vocabulary_size = 10
    embedding_size = 2
    num_sampled = 3  # Number of negative examples to sample.
    num_true = 1

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        train_dataset = tf.placeholder(tf.int32)
        train_labels = tf.placeholder(tf.int32, shape=[None, num_true])

        # Variables.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        # Model.
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        # Compute the softmax loss, using a sample of the negative labels each time.
        logits, labels = yba_sampled_softmax(model=model,
                                             weights=embeddings,
                                             inputs=embed,
                                             labels=train_labels,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size,
                                             num_true=num_true)
        loss = tf.reduce_mean(
            nn_ops.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=logits))

        # Optimizer.
        # Note: The optimizer will optimize the softmax_weights AND the embeddings.
        # This is because the embeddings are defined as a variable quantity and the
        # optimizer's `minimize` method will by default modify all variable quantities
        # that contribute to the tensor it is passed.
        # See docs on `tf.train.Optimizer.minimize()` for more details.
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        for step in range(100):
            # batch_data = np.array(
            #     [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0],
            #      [0, 1, 1, 0, 1], [0, 0, 0, 1, 0]])
            batch_data = np.array(
                [7, 7, 7, 7, 7])
            batch_labels = np.array([[1], [2], [3], [4], [5]])
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l, model_logits, model_labels, sampled_values = \
                session.run([
                    optimizer, loss,
                    model.logits,
                    model.labels,
                    model.sampled_values],
                    feed_dict=feed_dict)
            print('loss', l)
            print('sampled_values', sampled_values)
            if step == 99:
                x = 5


def _yba_sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
    # a matrix.  The gradient of _sum_rows(x) is more efficient than
    # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
    # we use _sum_rows(x) in the nce_loss() computation since the loss
    # is mostly used for training.
    cols = array_ops.shape(x)[1]
    ones_shape = array_ops.stack([cols, 1])
    ones = array_ops.ones(ones_shape, x.dtype)
    return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def _yba_compute_sampled_logits(model,
                                weights,
                                labels,
                                inputs,
                                num_sampled,
                                num_classes,
                                num_true=1,
                                sampled_values=None,
                                subtract_log_q=True,
                                remove_accidental_hits=False,
                                partition_strategy="mod",
                                name=None):
    """Helper function for nce_loss and sampled_softmax_loss functions.

    Computes sampled output training logits and labels suitable for implementing
    e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
    sampled_softmax_loss).

    Note: In the case where num_true > 1, we assign to each target class
    the target probability 1 / num_true so that the target probabilities
    sum to 1 per-example.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      subtract_log_q: A `bool`.  whether to subtract the log expected count of
          the labels in the sample to get the logits of the true labels.
          Default is True.  Turn off for Negative Sampling.
      remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  Default is
          False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).
    Returns:
      out_logits, out_labels: `Tensor` objects each with shape
          `[batch_size, num_true + num_sampled]`, for passing to either
          `nn.sigmoid_cross_entropy_with_logits` (NCE) or
          `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    """

    if not isinstance(weights, list):
        weights = [weights]

    with ops.name_scope(name, "yba_compute_sampled_logits",
                        weights + [inputs, labels]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        if sampled_values is None:
            sampled_values = candidate_sampling_ops.learned_unigram_candidate_sampler(
                true_classes=labels,
                num_true=num_true,
                num_sampled=num_sampled,
                unique=True,
                range_max=num_classes)
        model.sampled_values = sampled_values
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence
        sampled, true_expected_count, sampled_expected_count = sampled_values
        # pylint: enable=unpacking-non-sequence

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        all_ids = array_ops.concat([labels_flat, sampled], 0)

        # weights shape is [num_classes, dim]
        all_w = embedding_ops.embedding_lookup(
            weights, all_ids, partition_strategy=partition_strategy)
        # true_w shape is [batch_size * num_true, dim]
        true_w = array_ops.slice(
            all_w, [0, 0],
            array_ops.stack([array_ops.shape(labels_flat)[0], -1]))

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        dim = array_ops.shape(true_w)[1:2]
        new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
        row_wise_dots = math_ops.multiply(
            array_ops.expand_dims(inputs, 1),
            array_ops.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = array_ops.reshape(row_wise_dots,
                                           array_ops.concat([[-1], dim], 0))
        true_logits = array_ops.reshape(_yba_sum_rows(dots_as_matrix),
                                        [-1, num_true])

        # Lookup weights and biases for sampled labels.
        #   sampled_w shape is [num_sampled, dim]
        #   sampled_b is a [num_sampled] float tensor
        sampled_w = array_ops.slice(
            all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]),
            [-1, -1])

        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # sampled_b has shape [num_sampled]
        # Apply X*W', which yields [batch_size, num_sampled]
        sampled_logits = math_ops.matmul(
            inputs, sampled_w, transpose_b=True)

        if remove_accidental_hits:
            acc_hits = candidate_sampling_ops.compute_accidental_hits(
                labels, sampled, num_true=num_true)
            acc_indices, acc_ids, acc_weights = acc_hits

            # This is how SparseToDense expects the indices.
            acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
            acc_ids_2d_int32 = array_ops.reshape(
                math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
            sparse_indices = array_ops.concat(
                [acc_indices_2d, acc_ids_2d_int32], 1,
                "sparse_indices")
            # Create sampled_logits_shape = [batch_size, num_sampled]
            sampled_logits_shape = array_ops.concat(
                [array_ops.shape(labels)[:1],
                 array_ops.expand_dims(num_sampled, 0)],
                0)
            if sampled_logits.dtype != acc_weights.dtype:
                acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
            sampled_logits += sparse_ops.sparse_to_dense(
                sparse_indices,
                sampled_logits_shape,
                acc_weights,
                default_value=0.0,
                validate_indices=False)

        if subtract_log_q:
            # Subtract log of Q(l), prior probability that l appears in sampled.
            true_logits -= math_ops.log(true_expected_count)
            sampled_logits -= math_ops.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = array_ops.concat([true_logits, sampled_logits], 1)
        # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        # of ones. We then divide by num_true to ensure the per-example labels sum
        # to 1.0, i.e. form a proper probability distribution.
        out_labels = array_ops.concat([
            array_ops.ones_like(true_logits) / num_true,
            array_ops.zeros_like(sampled_logits)
        ], 1)

    return out_logits, out_labels


def yba_sampled_softmax(model,
                        weights,
                        labels,
                        inputs,
                        num_sampled,
                        num_classes,
                        num_true=1,
                        sampled_values=None,
                        remove_accidental_hits=True,
                        partition_strategy="mod",
                        name="yba_sampled_softmax_loss"):
    """Computes and returns the sampled softmax training loss.

    This is a faster way to train a softmax classifier over a huge number of
    classes.

    This operation is for training only.  It is generally an underestimate of
    the full softmax loss.

    At inference time, you can compute full softmax probabilities with the
    expression `tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)`.

    See our [Candidate Sampling Algorithms Reference]
    (../../extras/candidate_sampling.pdf)

    Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
    ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          [num_classes, dim].  The (possibly-sharded) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  Default is
          True.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).

    Returns:
      A `batch_size` 1-D tensor of per-example sampled softmax losses.

    """
    logits, labels = _yba_compute_sampled_logits(
        model=model,
        weights=weights,
        labels=labels,
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        subtract_log_q=True,
        remove_accidental_hits=remove_accidental_hits,
        partition_strategy=partition_strategy,
        name=name)
    model.logits = logits
    model.labels = labels
    # sampled_losses is a [batch_size] tensor.
    return logits, labels


def neg_sampling(input, batch, pos_labels, neg_labels, num_neg=5,
                 embed_dim=100):
    def generate_batch():
        return tf.reshape(tf.nn.embedding_lookup(input, batch),
                          shape=(-1, embed_dim, 1))

    def generate_samples():
        return tf.concat([tf.nn.embedding_lookup(input, pos_labels),
                          tf.nn.embedding_lookup(input, neg_labels)], 1)

    sims_col = num_neg + 1
    if FLAGS.need_second == 1:
        sims_col = 9

    sims = tf.reshape(tf.matmul(generate_samples(), generate_batch()), shape=(
        -1, sims_col))
    return sims


class NegSampler(object):
    def __init__(self, num_neg):
        self.ptr = 0
        self.num_neg = num_neg
        self.round = 0

    def init(self, N):
        self.li = list(range(N))
        if N % self.num_neg != 0:
            need = self.num_neg - N % self.num_neg
            self.li += list(range(need))
        assert (len(self.li) % self.num_neg == 0)

    def get_neg(self, pos_labels):
        cand_list = self.li[self.ptr:self.ptr + self.num_neg]
        s = (self.ptr + self.num_neg) % len(self.li)
        for i, cand in enumerate(cand_list):
            if cand in pos_labels:
                cand_list[i], s = self._find_next(pos_labels, s)
        return cand_list

    def increment(self):
        self.ptr = self.ptr + self.num_neg
        if self.ptr == len(self.li):
            self.ptr = 0
            shuffle(self.li)
            self.round += 1
        return self.round

    def _find_next(self, pos_labels, s):
        while True:
            cand = self.li[s]
            if cand not in pos_labels:
                return cand, (s + 1) % len(self.li)
            s = (s + 1) % len(self.li)


if __name__ == '__main__':
    # test()
    n = NegSampler(5)
    n.init(7)
    for i in range(4):
        for j in range(7):
            print(i, n.get_neg([0, 1, 2, 3]))
        n.increment()

