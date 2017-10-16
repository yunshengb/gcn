import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.preprocessing import normalize
import sys, os, datetime, collections
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

current_folder = os.path.dirname(os.path.realpath(__file__))


def prepare_exp_dir(flags):
    dir = 'exp/gcn_%s%s_%s' % (flags.dataset, '_%s' % flags.desc if flags.desc
    else '', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    intermediate_dir = '%s/intermediate' % dir
    logdir = '%s/log' % dir

    def makedir(dir):
        os.system('rm -rf %s && mkdir -pv %s' % (dir, dir))

    if not flags.debug:
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
    # adj = np.array(
    #     [[0,1,0],[1,0,1],[0,1,0]])
    adj = add_common_neighbor(adj)
    return load_data_from_adj(adj)


def load_blog_data():
    adj = np.load(
        '{}/data/BlogCatalog-dataset/data/blog_adj.npy'.format(current_folder))
    return load_data_from_adj(adj)


def load_flickr_data():
    def id(i):
        return int(i) - 1

    dic = collections.defaultdict(list)
    print('Loading flickr')
    with open('{}/data/Flickr-dataset/data/edges.csv'.format(c)) as f:
        for line in f:
            ls = line.rstrip().split(',')
            x = id(ls[0])
            y = id(ls[1])
            dic[x].append(y)
            dic[y].append(x)
    print('Loaded flickr')
    return load_data_from_adj(dic)

def load_arxiv_data():
    adj = np.load(
        '{}/data/arxiv/arxiv_adj.npy'.format(current_folder))
    adj = add_common_neighbor(adj)
    return load_data_from_adj(adj)

def add_common_neighbor(adj):
    N = adj.shape[0]
    larger = np.ones((N + 1, N + 1))
    larger[:-1,:-1] = adj
    np.fill_diagonal(larger, 0)
    return larger


def load_data_from_adj(adj):
    N = adj.shape[0] if type(adj) is not dict else len(adj)
    im = np.identity(N)
    labels = proc_labels(adj)
    labels_ = np.array(labels)
    features = sp.lil_matrix(im)
    return adj, features, labels, labels, labels


def load_data(dataset_str, embed):
    """Load data."""
    if dataset_str == 'syn':
        return load_synthetic_data()
    if dataset_str == 'blog':
        return load_blog_data()
    if dataset_str == 'flickr':
        return load_flickr_data()
    if dataset_str == 'arxiv':
        return load_arxiv_data()
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

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    if embed == 0:  # no embedding
        return adj, features, y_train, y_val, y_test
    else:
        labels = proc_labels(adj.todense())
        if embed == 1 or embed == 2:
            features = sp.lil_matrix(np.identity(labels.shape[0]))
        elif embed == 3:
            features = np.ones([labels.shape[0], 200])
        return select(adj), select(features), select(labels), select(labels), \
               select(labels)


def select(a, size=None):
    if not size:
        return a
    if a.ndim == 2:
        return a[0:size, 0:size]
    elif a.ndim == 1:
        return a[0:size]
    else:
        return None


def proc_labels(labels):
    # adj is dense.
    # zero_diagonal = np.ones(labels.shape)
    # np.fill_diagonal(zero_diagonal, 0)
    # labels = np.multiply(labels, zero_diagonal)
    for i in range(labels.shape[0]):
        # if np.count_nonzero(labels[i]) == 0:
        #     print('@@@@@')
        #     exit(1)
        if labels[i][i] != 0:
            print('#####')
            print(i)
            print(np.count_nonzero(labels[i]))
        #     exit(1)
        # print(np.count_nonzero(labels[i]))
    # print('Checked#######################')
    # return sparse_to_tuple(normalize_adj_sym(labels))
    # return normalize(labels, norm='l1')
    return normalize_adj_weighted_row(labels, weights=[0, 1.0, 0]).todense()


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj_sym(adj + sp.eye(adj.shape[0]))
    # adj_normalized = normalize_adj_row(adj + sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj_weighted_row(adj, weights=[0.7, 0.3,0])
    # x = np.array(normalize_adj_sym(adj + sp.eye(adj.shape[0])).todense())
    # y = np.array(normalize_adj_row(adj + sp.eye(adj.shape[0])).todense())
    # z = np.array(sp.coo_matrix(normalize_adj_2(adj.todense(), weights=[
    #         0.7, 0.3, 0])).todense())
    return sparse_to_tuple(adj_normalized)
    # return sparse_to_tuple(sp.eye(adj.shape[0]))


def normalize_adj_sym(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj is sparse.
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (adj.dot(d_mat_inv_sqrt).transpose().dot(
        d_mat_inv_sqrt).tocoo())

def normalize_adj_row(adj):
    # adj is sparse.
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt.dot(adj).tocoo())


def normalize_adj_weighted_row(adj, weights=[0.7, 0.2, 0.1]):
    if type(adj) is dict:
        return normalize_adj_weighted_row_from_dict(adj, weights)
    # adj is dense.
    def norm(neighbor, d, weight):
        return weight * normalize(np.multiply(neighbor, d), norm='l1')

    def div0(a, b):
        """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~ np.isfinite(c)] = 0  # -inf inf NaN
        return c

    one = adj
    self = np.identity(adj.shape[0])
    one_with_self = adj + self
    temp = one_with_self.dot(one_with_self)
    temp = div0(temp, temp)
    two = temp - one_with_self
    d = one.sum(1)
    normalized_adj = np.zeros(adj.shape)
    for i, neighbor in enumerate([self, one, two]):
        normalized_adj += norm(neighbor, d, weights[i])
    return (sp.coo_matrix(normalized_adj))


def normalize_adj_weighted_row_from_dict(neighbor_map, weights=[0.7,0.3,0]):
    print('@@@')
    N = len(neighbor_map)
    indices = []
    values = []
    shape = np.array([N, N], dtype=np.int64)
    indices += [(i, i) for i in range(N)]
    values += [weights[0] for i in range(N)]
    for i in range(N):
        norm = np.sum([len(neighbor_map[j]) for j in neighbor_map[i]])
        for j in neighbor_map[i]:
            indices.append((i, j))
            values.append(weights[1] * len(neighbor_map[j]) / norm)
    print('@@@')
    return indices, values, shape

def get_shape(mat):
    if type(mat) is tuple:
        return mat[2]
    return mat.shape

def construct_feed_dict(features, support, labels, placeholders,
                        embed):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
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


def print_var(sess, feed_dict, var, name, intermediate_dir, debug, save=False):
    if debug:
        return
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
