import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.preprocessing import normalize
import sys, os, datetime, collections
from random import shuffle
from math import ceil
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
    # adj = np.array(
    #     [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [0, 1, 1, 0, 1],
    #      [0, 0, 0, 1, 0]])
    # adj = np.array(
    #    [[0, 1, 1, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 0, 0, 0, 1],
    #    [0, 1, 1, 1, 0]])
    # adj = np.array(
    #     [[0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [1, 1, 1, 0, 0],
    #      [0, 1, 1, 0, 0]])
    # adj = np.array(
    #     [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 1],
    #      [0, 0, 0, 1, 0]])
    # adj = np.array(
    #     [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1],
    #      [0, 0, 0, 1, 0]])
    # adj = np.array(
    #     [[0,1,0],[1,0,1],[0,1,0]])
    # adj = add_common_neighbor(adj)
    # adj = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2, 4], 4: [3]}
    # adj = np.array(
    #     [[0, 1, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 0, 1],
    #      [1, 0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 0],
    #      [0, 0, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 1],
    #      [1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 1, 0]])
    # adj = np.array(
    #     [[0, 1, 1, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
    #      [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
    #      [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 1, 0]])
    # adj = np.array(
    #    [[0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 0, 0, 0, 1],
    #     [1, 0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 0],
    #     [0, 0, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 1],
    #     [1, 0, 0, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0]])
    adj = np.array(
        [[0, 1, 1, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 1, 0, 0],
         [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
         [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 1, 0]])
    # adj = np.array(
    #    [[0, 1, 1, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
    #     [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1],
    #     [0, 0, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
    #     [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0]])
    return load_data_from_adj(adj)


def load_blog_data(need_batch=True):
    labels = None
    if FLAGS.embed == 0:
        labels = np.load(
            '{}/data/BlogCatalog-dataset/data/blog_labels.npy'.format(
                current_folder))

    if not need_batch:
        adj = np.load(
            '{}/data/BlogCatalog-dataset/data/blog_adj.npy'.format(
                current_folder))
        return load_data_from_adj(adj, labels, need_batch=False)

    dic = collections.defaultdict(list)
    print('Loading blog')
    with open('{}/data/BlogCatalog-dataset/data/edges.csv'.format(
            current_folder)) as f:
        for line in f:
            ls = line.rstrip().split(',')
            x = id(ls[0])
            y = id(ls[1])
            dic[x].append(y)
            dic[y].append(x)
    dic = dict(dic)
    print('Loaded blog')

    return load_data_from_adj(dic, labels, need_batch=True)


def load_flickr_data():
    path = '{}/data/save/{}_neighbor_map.pickle'.format(current_folder,
                                                        FLAGS.dataset)
    dic = load(path)
    if not dic:
        dic = collections.defaultdict(list)
        print('Loading flickr')
        with open('{}/data/Flickr-dataset/data/edges.csv'.format(
                current_folder)) as f:
            for line in f:
                ls = line.rstrip().split(',')
                x = id(ls[0])
                y = id(ls[1])
                dic[x].append(y)
                dic[y].append(x)
        dic = dict(dic)
        print('Loaded flickr')
        save(path, dic)
    return load_data_from_adj(dic, need_batch=True)


def id(i):
    return int(i) - 1


def gen_hyper_neighbor_map(neighbor_map):
    rtn = collections.defaultdict(list)
    for i, nl in neighbor_map.items():
        rtn[len(nl)].append(i)
    return rtn


def load_arxiv_data():
    adj = np.load(
        '{}/data/arxiv/arxiv_cleaned_hidden.npy'.format(current_folder))
    # adj = add_common_neighbor(adj)
    return load_data_from_adj(adj)


def add_common_neighbor(adj):
    N = adj.shape[0]
    larger = np.ones((N + 1, N + 1))
    larger[:-1, :-1] = adj
    np.fill_diagonal(larger, 0)
    return larger


def load_data_from_adj(adj, labels=None, need_batch=False):
    N = get_shape(adj)[0]
    if labels is None:
        labels = proc_labels(adj)
    else:
        labels = normalize(labels, norm='l1')
    features = None
    train_mask = sample_mask(range(N), N)
    test_ids = list(range(N))
    shuffle(test_ids)
    test_ids = test_ids[0:ceil(0.1 * N)]
    return adj, features, labels, train_mask, test_ids, need_batch


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

    train_mask = sample_mask(idx_train, labels.shape[0])

    if embed == 0:  # no embedding
        return adj, features, y_train, train_mask, test_idx_range, False
    else:
        labels = proc_labels(adj.todense())
        if embed == 1 or embed == 2:
            features = sp.lil_matrix(np.identity(labels.shape[0]))
        elif embed == 3:
            features = np.ones([labels.shape[0], 200])
        return select(adj.todense()), select(features), select(
            labels), train_mask, \
               test_idx_range, False


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
    # for i in range(labels.shape[0]):
    #     # if np.count_nonzero(labels[i]) == 0:
    #     #     print('@@@@@')
    #     #     exit(1)
    #     if labels[i][i] != 0:
    #         print('#####')
    #         print(i)
    #         print(np.count_nonzero(labels[i]))
    #         #     exit(1)
    #         # print(np.count_nonzero(labels[i]))
    # print('Checked#######################')
    # return sparse_to_tuple(normalize_adj_sym(labels))
    # return normalize(labels, norm='l1')
    if type(labels) is dict:
        return labels
    else:
        # return normalize(labels, norm='l1')
        return normalize_adj_weighted_row(labels, weights=[0, 1,
                                                           0]).todense()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj_sym(adj + sp.eye(adj.shape[0]))
    # adj_normalized = normalize_adj_row(adj + sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj_weighted_row(adj, weights=[0.7, 0.3, 0])
    # x = np.array(normalize_adj_sym(adj + sp.eye(adj.shape[0])).todense())
    # y = np.array(normalize_adj_row(adj + sp.eye(adj.shape[0])).todense())
    # z = np.array(sp.coo_matrix(normalize_adj_2(adj.todense(), weights=[
    #         0.7, 0.3, 0])).todense())
    if type(adj_normalized) is tuple:
        return adj_normalized
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


def normalize_adj_weighted_row(adj, weights=[0.7, 0.2, 0.1], inverse=True):
    if type(adj) is dict:
        return normalize_adj_weighted_row_from_dict(adj, weights, inverse)

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
    if inverse:
        d = 1. / d
    normalized_adj = np.zeros(adj.shape)
    for i, neighbor in enumerate([self, one, two]):
        normalized_adj += norm(neighbor, d, weights[i])
    print('normalized_adj', normalized_adj)
    return (sp.coo_matrix(normalized_adj))


def normalize_adj_weighted_row_from_dict(neighbor_map, weights=[0.7, 0.3, 0],
                                         inverse=True):
    print('@@@ normalize_adj_weighted_row_from_dict')
    path = '{}/data/save/{}_weighted_row_norm.pickle'.format(current_folder,
                                                             FLAGS.dataset)
    rtn = load(path)
    if rtn:
        return rtn
    N = len(neighbor_map)
    indices = []
    values = []
    shape = np.array([N, N], dtype=np.int64)
    indices += [(i, i) for i in range(N)]
    values += [weights[0] for i in range(N)]
    for i in range(N):
        norm = np.sum([len(neighbor_map[j]) for j in neighbor_map[i]])
        if inverse:
            norm = np.sum([1 / len(neighbor_map[j]) for j in neighbor_map[i]])
        for j in neighbor_map[i]:
            indices.append((i, j))
            values.append(weights[1] * len(neighbor_map[j]) / norm)
    print('@@@ normalize_adj_weighted_row_from_dict done')
    save(path, (indices, values, shape))
    return indices, values, shape


def normalize_batch_labels_weighted_row_from_dict(neighbor_map, batch,
                                                  weights=[0, 1, 0],
                                                  inverse=True):
    def get_norm(neighbor_map, i, inverse):
        if inverse:
            return 1 / len(neighbor_map[i])
        else:
            return len(neighbor_map[i])

    M = len(batch)
    N = len(neighbor_map)
    mat = np.zeros((M, N))
    for i, real_id in enumerate(batch):
        norm = np.sum([get_norm(neighbor_map, j, inverse) for j in neighbor_map[
            real_id]])
        for j in neighbor_map[real_id]:
            mat[i][j] = weights[1] * get_norm(neighbor_map, j, inverse) / norm
    return mat


def load(fn):
    if not os.path.exists(fn):
        return None
    with open(fn, 'rb') as handle:
        rtn = pkl.load(handle)
    print('Loaded {}'.format(fn))
    return rtn


def save(fn, obj):
    with open(fn, 'wb') as handle:
        pkl.dump(obj, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('saved {}'.format(fn))


def get_shape(mat):
    if type(mat) is tuple:
        return mat[2]
    elif type(mat) is dict:
        # Neighbor map.
        return (len(mat),)
    return mat.shape


def construct_feed_dict(adj, support, labels, labels_mask, placeholders,
                        embed, need_batch):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update(
        {placeholders['support'][i]: support[i] for i in range(len(support))})
    if need_batch:
        assert (type(labels) is dict)
        batch, batch_labels, sims_mask = generate_batch(adj)
        feed_dict.update({placeholders['batch']: batch})
        feed_dict.update({placeholders['labels']: batch_labels})
        feed_dict.update({placeholders['sims_mask']: sims_mask})
    else:
        feed_dict.update({placeholders['labels']: labels})
        if embed == 2:
            N = get_shape(labels)[0]
            inf_diagonal = np.zeros((N, N))
            np.fill_diagonal(inf_diagonal, 999999999999999999)
            feed_dict.update(
                {placeholders['sims_mask']: np.ones((N, N)) - np.identity(
                    N) - inf_diagonal})
    return feed_dict


data_index = 0
round = 0
batch_size = 10312//2


def generate_batch(neighbor_map):
    global data_index, round
    N = len(neighbor_map)
    end = data_index + batch_size
    if end >= N:
        end = N
    batch = list(range(data_index, end))
    M = len(batch)
    print('batch_size %s \tround %s' % (M, round))
    rtn_labels = normalize_batch_labels_weighted_row_from_dict(neighbor_map,
                                                               batch)
    sims_mask = np.ones((M, N))
    for i, j in enumerate(batch):
        sims_mask[i][j] = -999999999999999999
    data_index = end
    if data_index == N:
        data_index = 0
        round += 1
    return batch, rtn_labels, sims_mask


# data_index = 0
# round = 0
#
#
# def generate_batch(neighbor_map, hyper_neighbor_map):
#     global data_index, round
#     # batch_size, num_data = get_size(neighbor_map, data_index, max_size)
#     size_li = list(hyper_neighbor_map.keys())
#     num_true = size_li[data_index]
#     data_li = hyper_neighbor_map[num_true]
#     batch_size = len(data_li)
#     print('num_true %s \tbatch_size %s \tround %s' % (num_true, batch_size,
#                                                     round))
#     batch = np.array(data_li)
#     labels = np.ndarray(shape=(batch_size, num_true), dtype=np.int32)
#     for i, id in enumerate(data_li):
#         true_neighbors = neighbor_map[id]
#         assert(len(true_neighbors) == num_true)
#         labels[i] = true_neighbors
#     data_index += 1
#     if data_index + 1 == len(hyper_neighbor_map):
#         round += 1
#         data_index = 0
#     return batch, labels, num_true

# max_size = 667969
# def get_size(neighbor_map, data_index, max_size):
#     size = 0
#     i = data_index
#     cnt = 0
#     while size + len(neighbor_map[i]) <= max_size:
#         size += len(neighbor_map[i])
#         i = incr_index(neighbor_map, i)
#         cnt += 1
#     return size, cnt
#
#
# def incr_index(neighbor_map, idx):
#     return get_id(neighbor_map, idx + 1)
#
#
# def get_id(neighbor_map, idx):
#     return idx % len(neighbor_map)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj_sym(adj)
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


def print_var(var, name, intermediate_dir, sess=None, feed_dict=None):
    # Output variable.
    if sess and feed_dict:
        var = sess.run(var, feed_dict=feed_dict)
    if not FLAGS.debug:
        fn = '%s/%s.npy' % (intermediate_dir, name)
        print('%s dumped to %s with shape %s' % (name, fn, var.shape))
        np.save(fn, var)
    else:
        print(name)
        print(var)
