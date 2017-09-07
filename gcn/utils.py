import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.preprocessing import normalize
import sys, os, datetime

current_folder = os.path.dirname(os.path.realpath(__file__))


def prepare_exp_dir(flags):
    dir = 'gcn_%s%s_%s' % (flags.dataset, '_%s' % flags.desc if flags.desc
    else '', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    intermediate_dir = '%s/intermediate' % dir
    logdir = '%s/log' % dir
    def makedir(dir):
        os.system('rm -rf %s && mkdir -pv %s' % (dir, dir))
    makedir(intermediate_dir)
    makedir(logdir)
    return dir, intermediate_dir, logdir


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_synthetic_data():
    adj = np.array(
        [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [0, 1, 1, 0, 1],
         [0, 0, 0, 1,
          0]])
    im = np.identity(adj.shape[0])
    labels = normalize(adj + im, norm='l1')
    features = sp.lil_matrix(im)
    idx_test = range(len(labels))
    idx_train = range(len(labels))
    idx_val = range(len(labels))
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    return adj, features, labels, labels, labels, train_mask, val_mask, test_mask


def load_data(dataset_str, embed):
    """Load data."""
    if dataset_str == 'syn':
        return load_synthetic_data()
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/data/ind.{}.{}".format(current_folder, dataset_str,
                                             names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "{}/data/ind.{}.test.index".format(current_folder, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    if embed != 0:
        idx_test = range(len(labels))
        idx_train = range(len(labels))
        idx_val = range(len(labels))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    if embed == 0:  # no embedding
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    else:
        im = np.identity(labels.shape[0])
        labels = adj.todense()
        if embed == 1 or embed == 2:
            features = sp.lil_matrix(im)
        elif embed == 3:
            features = np.ones([labels.shape[0], 200])
        # return adj, features, labels, labels, labels, train_mask, val_mask, test_mask
        # x = labels[3918]
        # print(np.count_nonzero(x))
        # print(np.extract(x!=0, x))
        # print(x[3917])
        zero_diagonal = np.ones(labels.shape)
        np.fill_diagonal(zero_diagonal, 0)
        labels = np.multiply(labels, zero_diagonal)
        labels = normalize(labels, norm='l1')
        return select(adj), select(features), select(labels), select(labels), \
        select(labels), select(train_mask), select(val_mask), select(test_mask)


def select(a, size=None):
    if not size:
        return a
    if a.ndim == 2:
        return a[0:size, 0:size]
    elif a.ndim == 1:
        return a[0:size]
    else:
        return None


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders,
                        embed):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update(
        {placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    if embed == 2:
        inf_diagonal = np.zeros(labels.shape)
        np.fill_diagonal(inf_diagonal, 999999999999999999)
        feed_dict.update(
            {placeholders['sims_mask']: np.ones(labels.shape) - np.identity(
                labels.shape[0]) - inf_diagonal})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(
        adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def print_var(sess, feed_dict, var, name, intermediate_dir, save=False):
    # Output variable.
    var = sess.run([var], feed_dict=feed_dict)
    assert (len(var) == 1)
    var = var[0]
    if not save:
        print(name)
        print(var)
    if save:
        fn = '%s/%s.npy' % (intermediate_dir, name)
        print('%s dumped to %s with shape %s' % (name, fn, var.shape))
        np.save(fn, var)
