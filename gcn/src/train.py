from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP

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
flags.DEFINE_string('desc',
                    'ns_sym',
                    'Description of the '
                    'experiment.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10001, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden layer 2.')
# flags.DEFINE_integer('hidden3', 100, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('embed', 0, '0: No embedding; 1|2.')
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
adj, features, y_train, train_mask, test_ids, need_batch = \
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
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
N = get_shape(adj)[0]
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'dropout': tf.placeholder_with_default(0., shape=()),
    'output_dim': get_shape(y_train),
}

if FLAGS.embed == 0:
    placeholders['train_mask'] = tf.placeholder(tf.int32, shape=(N,))
    placeholders['labels'] = tf.placeholder(tf.float32, shape=(None, y_train.shape[1]))

if need_batch:
    placeholders['batch'] = tf.placeholder(tf.int32)
    placeholders['pos_labels'] = tf.placeholder(tf.int32)
    placeholders['neg_labels'] = tf.placeholder(tf.int32)
    if FLAGS.embed != 0:
        placeholders['labels'] = tf.placeholder(tf.int32)
    placeholders['num_data'] = get_shape(adj)[0]
else:
    if FLAGS.embed == 2:
        placeholders['labels'] = tf.placeholder(tf.float32, shape=(N, N))
        placeholders['sims_mask'] = tf.placeholder(tf.float32,
                                                   shape=(N, N))

# Create model
model = model_func(placeholders, input_dim=N, logging=True)

# Initialize session
session_conf = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 0},
    allow_soft_placement=True,
    log_device_placement=False
)

if FLAGS.dataset == 'flickr':
    sess = tf.Session(config=session_conf)
else:
    sess = tf.Session()


def need_print(epoch=None):
    # return False
    if not epoch:
        return True
    # return epoch < 50 or epoch % 5 == 0
    return epoch % 100 == 0
    # return False


# Summary.
dir, intermediate_dir, logdir = prepare_exp_dir(FLAGS)
merged = tf.summary.merge_all()
if need_print():
    train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(logdir + '/test')

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj, support, y_train, train_mask,
                                    placeholders, FLAGS.embed, need_batch)

    if need_print(epoch):
        if FLAGS.embed > 0:
            print_var(model.layers[-1].embeddings,
                      'gcn_%s_emb_%s' % (FLAGS.dataset, epoch),
                      intermediate_dir, sess, feed_dict)
            if epoch == 0:
                print_var(np.array(test_ids),
                          'gcn_%s_tids' % (FLAGS.dataset),
                          intermediate_dir)
        print_var(model.loss,
                  'gcn_%s_loss_%s' % (FLAGS.dataset, epoch), intermediate_dir,
                  sess, feed_dict)

    # Training step
    fetches = [model.opt_op, model.loss, merged]
    if FLAGS.embed == 0:
        fetches.append(tf.nn.embedding_lookup(model.layers[-1].outputs, test_ids))
    if need_print(epoch):
        fetches.append(merged)
    # print('###placeholders', placeholders)
    # print('$$$feed_dict', feed_dict)
    outs = sess.run(fetches, feed_dict=feed_dict)

    # if np.abs(outs[1] - 1.61800) < 0.0001:
    #    print_var(model.layers[-1].embeddings,
    #              '~~~~~gcn_%s_emb_%s' % (FLAGS.dataset, epoch),
    #              intermediate_dir, sess, feed_dict)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
          "{:.5f}".format(outs[1]),
          "time=",
          "{:.5f}".format(time.time() - t))
    if FLAGS.embed == 0:
        preds = outs[3]
        print('preds', preds.shape)


    if need_print(epoch):
        train_writer.add_summary(outs[2], epoch)

print("Optimization Finished!")
