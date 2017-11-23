from __future__ import division
from __future__ import print_function

from metrics import masked_accuracy
import time
import tensorflow as tf
import numpy as np

from utils import *
from models import GCN

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'blog', 'Dataset string.')
# 'cora', 'citeseer', 'pubmed', 'syn', 'blog', 'flickr', 'arxiv'
flags.DEFINE_integer('debug', 1, '0: Normal; 1: Debug.')
flags.DEFINE_string('model', 'gcn',
                    'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('desc', 'joint_weighted_0_7_0_3_inverse',
                    'Description of the experiment.')
flags.DEFINE_integer('need_batch', 1, 'Need min-batch or not.')
flags.DEFINE_string('device', 'cpu', 'cpu|gpu.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10001, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 50, 'Number of units in hidden layer 3.')
flags.DEFINE_float('train_ratio', 0.1, 'Ratio of training over testing data.')
flags.DEFINE_integer('embed', 0, '0: No embedding; 1|2|3.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4,
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, train_mask, test_ids = \
    load_data(FLAGS.dataset, FLAGS.embed)

# Some preprocessing
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
N = get_shape(adj)[0]
placeholders = {
    #'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support':None,
    # 'dropout': None,
    'output_dim': get_shape(y_train),
}

if FLAGS.embed == 0 or FLAGS.embed == 3:
    if features is not None:
        placeholders['features'] = tf.placeholder(tf.float32, shape=(None, features.shape[1]))
    placeholders['train_mask'] = tf.placeholder(tf.int32, shape=(N,))
    placeholders['ssl_labels'] = tf.placeholder(tf.float32,
                                            shape=(None, y_train.shape[1]))

if FLAGS.need_batch and (FLAGS.embed == 2 or FLAGS.embed == 3):
    placeholders['batch'] = tf.placeholder(tf.int32)
    placeholders['pos_labels'] = tf.placeholder(tf.int32)
    placeholders['neg_labels'] = tf.placeholder(tf.int32)
    placeholders['usl_labels'] = tf.placeholder(tf.int32, shape=(None,6))
    placeholders['num_data'] = get_shape(adj)[0]
elif FLAGS.embed == 2:
    placeholders['labels'] = tf.placeholder(tf.float32, shape=(N, N))
    placeholders['sims_mask'] = tf.placeholder(tf.float32,
                                               shape=(N, N))

# Create model
input_dim = features.shape[1] if features is not None else N
model = model_func(placeholders, input_dim=input_dim, logging=True)

# Initialize session
session_conf = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 0},
    allow_soft_placement=True,
    log_device_placement=False
)

if FLAGS.device == 'cpu':
    sess = tf.Session(config=session_conf)
else:
    sess = tf.Session()


def need_print(epoch=None):
    if FLAGS.debug or not epoch:
        return False
    return epoch % 50 == 0

def get_mode(epoch):
    if FLAGS.embed == 0:
        return 0
    elif FLAGS.embed == 2:
        return 2
    elif FLAGS.embed == 3:
        if epoch % 2 == 0:
            return 2
        else:
            return 0


# Summary.
dir = prepare_exp_dir(FLAGS)

# Init variables
sess.run(tf.global_variables_initializer())

f1_micros, f1_macros = [], []

# Train model
for epoch in range(FLAGS.epochs):
    mode = get_mode(epoch)
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask,
                                    placeholders, mode)
    # feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    if need_print(epoch):
        if FLAGS.embed > 0:
            embeddings = model.layers[-1].outputs if FLAGS.embed == 3 else \
                model.layers[-1].outputs
            print_var(embeddings,
                      'gcn_%s_emb_%s' % (FLAGS.dataset, epoch),
                      dir, sess, feed_dict)
        print_var(model.loss,
                  'gcn_%s_loss_%s' % (FLAGS.dataset, epoch), dir,
                  sess, feed_dict)

    # Training step
    if FLAGS.embed == 0 or FLAGS.embed == 2:
        fetches = [model.opt_op, model.loss]
    elif FLAGS.embed == 3:
        if mode == 0:
            fetches = [model.ssl_opt_op, model.ssl_loss]
        elif mode == 2:
            fetches = [model.usl_opt_op, model.usl_loss]
    if mode == 0:
        preds = model.ssl_layers[-1].outputs if FLAGS.embed == 3 else \
            model.outputs
        fetches.append(tf.nn.embedding_lookup(preds,
                                              test_ids))
        fetches.append(tf.nn.embedding_lookup(y_train, test_ids))
    outs = sess.run(fetches, feed_dict=feed_dict)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
          "{:.5f}".format(outs[1]),
          "time=",
          "{:.5f}".format(time.time() - t))
    if mode == 0:
        y_preds = outs[2]
        y_labels = outs[3]
        y_labels[y_labels > 0] = 1
        f1_micro, f1_macro = masked_accuracy(y_preds, y_labels)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
        print('f1_micro, f1_macro', f1_micro, f1_macro)
        print('max f1_micros, max f1_macros', np.max(f1_micros), np.max(
            f1_macros))


print("Optimization Finished!")
