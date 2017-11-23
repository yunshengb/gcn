import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import scipy.io
import os, sys

c_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(c_folder, "../../src"))
print(os.path.join(c_folder, "../../src"))
from utils import load_data
from metrics import masked_softmax_cross_entropy, masked_accuracy
from layers import Dense


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'blog', 'Dataset string.')
flags.DEFINE_integer('embed', 0, '0: No embedding; 1|2.')
flags.DEFINE_integer('dim', 100, 'Embedding dimension.')
flags.DEFINE_float('train_ratio', 0.1, 'Ratio of training over testing data.')
flags.DEFINE_integer('need_batch', 1, 'Need min-batch or not.')

# E = '/home/yba/Documents/gcn/gcn/analysis/node_classification/blog_100d/blog_emb_iter_1_p_0.25_q_0.25_walk_40_win_10.npy'
E = 'blog_emb_iter_1_p_0.25_q_0.25_walk_40_win_10.npy'
# E = 'gcn_blog_emb_6100.npy'
# E = 'blog_100d_embedding.mat'
# E = 'line_blog_unabridged_100.npy'
# E = '/home/yba/Documents/gcn/gcn/src/../exp/gcn_blog_joint_weighted_0_7_0_3_inverse_20171122225031/gcn_blog_emb_150.npy'

def main():
    y_train, train_mask, test_ids, embedding = load()
    N = y_train.shape[0]
    M = y_train.shape[1]
    placeholders, opt_op, loss_, preds_ = construct(M, N)
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess = tf.Session(config=session_conf)
    sess.run(tf.global_variables_initializer())
    f1_micros = []
    f1_macros = []
    for i in range(500):
        feed_dict = {
            placeholders['embedding']: embedding,
            placeholders['train_mask']: train_mask,
            placeholders['labels']: y_train
        }
        fetches = [opt_op, loss_, tf.nn.embedding_lookup(preds_, test_ids),
                         tf.nn.embedding_lookup(y_train, test_ids)]
        f1_micro, f1_macro = train_loop(sess, fetches, feed_dict, i)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
        print('max f1_micros, max f1_macros', np.max(f1_micros), np.max(
            f1_macros))


def load():
    adj, features, y_train, train_mask, test_ids = \
        load_data(FLAGS.dataset, 0)
    print(y_train.shape)
    if 'npy' in E:
        embedding = np.load(E)
    elif 'mat' in E:
        embedding = scipy.io.loadmat(E)['embedding']
    return y_train, train_mask, test_ids, embedding


def construct(M, N):
    # placeholders = {
    #     'embedding': tf.placeholder(tf.float32, shape=(None, FLAGS.dim)),
    #     'train_mask': tf.placeholder(tf.int32, shape=(N,)),
    #     'labels': tf.placeholder(tf.float32, shape=(None, M))
    #
    # }
    placeholders = {
        # 'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support_': None,
        'output_dim': 39,
    }

    if FLAGS.embed == 0 or FLAGS.embed == 3:
        placeholders['embedding'] = tf.placeholder(tf.float32, shape=(None, 100))
        placeholders['train_mask'] = tf.placeholder(tf.int32, shape=(N,))
        placeholders['labels'] = tf.placeholder(tf.float32,
                                                    shape=(None, 39))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    preds = fully_connected(placeholders['embedding'], M, activation_fn=lambda x: x)
    loss = masked_softmax_cross_entropy(preds,
                                        placeholders['labels'],
                                        placeholders['train_mask'])
    opt_op = optimizer.minimize(loss)
    return placeholders, opt_op, loss, preds

def train_loop(sess, fetches, feed_dict, epoch):
    outs = sess.run(fetches, feed_dict=feed_dict)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
          "{:.5f}".format(outs[1]))
    preds = outs[2]
    labels = outs[3]
    labels[labels > 0] = 1
    f1_micro, f1_macro = masked_accuracy(preds, labels)
    print('f1_micro, f1_macro', f1_micro, f1_macro)
    return f1_micro, f1_macro


if __name__ == '__main__':
    main()
