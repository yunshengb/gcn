from __future__ import division
from __future__ import print_function

import time, os
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
# 'cora', 'citeseer', 'pubmed', 'syn', 'blog', 'flickr
flags.DEFINE_integer('debug', 0, '0: Normal; 1: Debug.')
flags.DEFINE_string('model', 'gcn',
                    'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('desc', 'sym_norm', 'Description of the '
                                                'experiment.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 201, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 400, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 200, 'Number of units in hidden layer 2.')
#flags.DEFINE_integer('hidden3', 100, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('embed', 2, '0: No embedding; 1|2|3.')
# Plan 1: Dense layer after conv
# Plan 2: Embedding layer conv
# Plan 3: No conv
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4,
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# PRINT_EPOCHES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
#                  17, 18, 19, 20, 30, 40, 50,
#                  60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]
# PRINT_EPOCHES = [0, 10, 20, 30, 40, 50,
#                  60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]
# PRINT_EPOCHES = []

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
    load_data(FLAGS.dataset, FLAGS.embed)

# name = '%s_truth' % FLAGS.dataset
# fn = '%s.npy' % name
# print('%s dumped to %s with shape %s' % (name, fn, y_train.shape))
# np.save(fn, y_train)
# exit(1)

# Some preprocessing
if FLAGS.embed != 3:
    features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2],
                                                                    dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)
    # helper variable for sparse dropout
}
if FLAGS.embed == 3:
    placeholders['features'] = tf.placeholder(tf.float32, shape=(
        y_train.shape[0], FLAGS.hidden2))
if FLAGS.embed == 2:
    placeholders['sims_mask'] = tf.placeholder(tf.float32, shape=y_train.shape)

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()

def need_print(epoch=None):
    if FLAGS.debug:
        return False
    if not epoch:
        return True
    return epoch < 50 or epoch % 5 == 0
    # return False

# Summary.
dir, intermediate_dir, logdir = prepare_exp_dir(FLAGS)
merged = tf.summary.merge_all()
if need_print():
    train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(logdir + '/test')


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask,
                                        placeholders, FLAGS.embed)
    outs_val = sess.run([model.loss, model.accuracy, merged],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask,
                                    placeholders, FLAGS.embed)
    # feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    if need_print(epoch):
        print_var(sess, feed_dict, model.layers[-1].embeddings,
                  'gcn_%s_emb_%s' % (FLAGS.dataset, epoch), intermediate_dir,
                  FLAGS.debug, True)
        print_var(sess, feed_dict, model.loss,
                  'gcn_%s_loss_%s' % (FLAGS.dataset, epoch), intermediate_dir,
                  FLAGS.debug, True)

    # Training step
    fetches = [model.opt_op, model.loss, model.accuracy, merged]
    if need_print(epoch):
        fetches.append(merged)
    outs = sess.run(fetches, feed_dict=feed_dict)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
          "{:.5f}".format(outs[1]),
          "time=",
          "{:.5f}".format(time.time() - t))

    if need_print(epoch):
        train_writer.add_summary(outs[-1], epoch)

print("Optimization Finished!")

# Testing
test_cost, test_acc, summary, test_duration = evaluate(features, support,
                                                       y_test,
                                              test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "time=", "{:.5f}".format(test_duration))
if FLAGS.embed == 0:
    print("Accuracy={:.5f}".format(test_acc))

if need_print():
    test_writer.add_summary(summary, FLAGS.epochs-1)
