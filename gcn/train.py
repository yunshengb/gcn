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
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora','citeseer',
# 'pubmed','syn
flags.DEFINE_string('model', 'gcn',
                    'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 200, 'Number of units in hidden layer 2.')
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

# PRINT_EPOCHES = [0, 10, 20, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 50,
#                  60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]
PRINT_EPOCHES = [0]

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
    load_data(FLAGS.dataset, FLAGS.embed)

# name = 'y_train'
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
    placeholders['features'] = tf.placeholder(tf.float32, shape=(2708, 200))

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask,
                                        placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

os.system('rm -rf intermediate && mkdir intermediate')

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask,
                                    placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    if False:
        print()
        # print_var(sess, feed_dict, model.activations[0],
        #           'input')
        # print_var(sess, feed_dict, model.activations[1],
        #           'output of cv1')
        print_var(sess, feed_dict, model.layers[-1].vars['embed_mask'],
                  'embed_mask')
        # print_var(sess, feed_dict, model.layers[-1].embeddings, 'embeddings')
        print_var(sess, feed_dict, model.layers[-1].output, 'output')
        print_var(sess, feed_dict, model.preds, 'preds')
        print_var(sess, feed_dict, model.probs, 'probs')
    if epoch == FLAGS.epochs - 1:
        print_var(sess, feed_dict, model.layers[-1].embeddings, 'cora_embed',
                  True)

    # Training step
    fetches = [model.opt_op, model.loss, model.accuracy]
    if epoch in PRINT_EPOCHES:
        fetches.append(model.printer)
    outs = sess.run(fetches, feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask,
                                   placeholders)
    cost_val.append(cost)

    # if FLAGS.embed != 0:
    #     eq = tf.assert_equal(y_train, y_val)
    #     sess.run([eq], feed_dict={placeholders['labels']: y_train, placeholders['labels']: y_val})
    #     eq = tf.assert_equal(train_mask, val_mask)
    #     sess.run([eq], feed_dict={placeholders['labels_mask']: train_mask, placeholders['labels_mask']: val_mask})

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
          "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=",
          "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=",
          "{:.5f}".format(time.time() - t))

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test,
                                              test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=",
      "{:.5f}".format(test_duration))


