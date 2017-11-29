from __future__ import division
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.preprocessing import normalize
import sys, os, datetime, collections
import random
from math import ceil
import tensorflow as tf
from neg_sampling import NegSampler

flags = tf.app.flags
FLAGS = flags.FLAGS

current_folder = os.path.dirname(os.path.realpath(__file__))


def prepare_exp_dir(flags):
    dir = '%s/../exp/gcn_%s%s_%s' % (current_folder, flags.dataset, '_%s' %
                                                                    flags.desc if
    flags.desc
    else '', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    def makedir(dir):
        os.system('rm -rf %s && mkdir -pv %s' % (dir, dir))

    if not flags.debug:
        makedir(dir)
    return dir


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


def load_blog_data():
    labels = None
    features = None
    # features = np.load(
    #     '{}/../data/BlogCatalog-dataset/data/blog_emb_iter_1_p_0.25_q_0.25_walk_40_win_10.npy'.format(
    #         current_folder))
    if FLAGS.embed == 0 or FLAGS.embed == 3:
        labels = np.load(
            '{}/../data/BlogCatalog-dataset/data/blog_labels.npy'.format(
                current_folder))

    if not FLAGS.need_batch:
        adj = np.load(
            '{}/../data/BlogCatalog-dataset/data/blog_adj.npy'.format(
                current_folder))
        return load_data_from_adj(adj, labels, features)

    dic = collections.defaultdict(list)
    print('Loading blog')
    with open('{}/../data/BlogCatalog-dataset/data/edges.csv'.format(
            current_folder)) as f:
        for line in f:
            ls = line.rstrip().split(',')
            x = id(ls[0])
            y = id(ls[1])
            dic[x].append(y)
            dic[y].append(x)
    dic = dict(dic)
    print('Loaded blog')

    return load_data_from_adj(dic, labels, features)


def load_flickr_data():
    path = '{}/../data/save/{}_neighbor_map.pickle'.format(current_folder,
                                                        FLAGS.dataset)
    dic = load(path)
    if not dic:
        dic = collections.defaultdict(list)
        print('Loading flickr')
        with open('{}/../data/Flickr-dataset/data/edges.csv'.format(
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
    return load_data_from_adj(dic)


def id(i):
    return int(i) - 1


def gen_hyper_neighbor_map(neighbor_map):
    rtn = collections.defaultdict(list)
    for i, nl in neighbor_map.items():
        rtn[len(nl)].append(i)
    return rtn


def load_arxiv_data():
    adj = np.load(
        '{}/../data/arxiv/arxiv_cleaned_hidden.npy'.format(current_folder))
    # adj = add_common_neighbor(adj)
    return load_data_from_adj(adj)


def add_common_neighbor(adj):
    N = adj.shape[0]
    larger = np.ones((N + 1, N + 1))
    larger[:-1, :-1] = adj
    np.fill_diagonal(larger, 0)
    return larger

#add mask
def load_data_from_adj(adj, labels=None, features=None):
    N = get_shape(adj)[0]
    if labels is None:
        remove_self = True if FLAGS.dataset == 'arxiv' else False
        labels = proc_labels(adj, remove_self)
    else:
        labels = normalize(labels, norm='l1')
    # if features is not None:
    #     features = normalize(features, norm='l2')
    train_mask = sample_mask(range(N), N)
    all_ids = list(range(N))
    random.Random(123).shuffle(all_ids)
    valid_ids = all_ids[0:ceil(0.1 * N)] # 10%
    test_ids = all_ids[ceil(0.1 * N):ceil((1 - FLAGS.train_ratio) * N)] # test 1-train_ratio-0.1
    train_mask.fill(1)
    unseen_ids = (valid_ids + test_ids)
    for id in unseen_ids:
        train_mask[id] = 0
    return adj, features, labels, train_mask, valid_ids, test_ids


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
        with open("{}/../data/ind.{}.{}".format(current_folder, dataset_str,
                                             names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "{}/../data/ind.{}.test.index".format(current_folder, dataset_str))
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


def proc_labels(labels, remove_self=False):
    # adj is dense.
    if remove_self:
        zero_diagonal = np.ones(labels.shape)
        np.fill_diagonal(zero_diagonal, 0)
        labels = np.multiply(labels, zero_diagonal)
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
        return normalize(labels, norm='l1')
        #return normalize_adj_weighted_row(labels, weights=[0, 1,
        #                                                   0],
        #                                  inverse=True).todense()


def preprocess_adj(adj):#laplacian
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj_sym(adj + sp.eye(adj.shape[0]))
    #adj_normalized = normalize_adj_row(adj + sp.eye(adj.shape[0]))
    #adj_normalized = normalize_adj_weighted_row(adj, weights=[0.58, 0.42, 0])
    #                                            inverse=False)
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


def get_norm(neighbor_map, i, inverse, pp=False):
    if inverse:
        if pp:
            print('{} {} {}'.format(1, len(neighbor_map[i]) , 1 / len(neighbor_map[i])))
        return 1 / len(neighbor_map[i])
    else:
        return len(neighbor_map[i])


def normalize_adj_weighted_row_from_dict(neighbor_map, weights=[0.7, 0.3, 0],
                                         inverse=True):
    print('@@@ normalize_adj_weighted_row_from_dict')
    path = '{}/../data/save/{}_weighted_row_norm_{}_{}.pickle'.format(
        current_folder,
        FLAGS.dataset,
        str(weights), \
        inverse)
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
        norm = np.sum(
            [get_norm(neighbor_map, j, inverse) for j in neighbor_map[i]])
        if norm == 0:
            print('{} neighbor {} has norms {}'.format(i, neighbor_map[i], [get_norm(neighbor_map, j, inverse, pp=True) for j in neighbor_map[i]]))
        for j in neighbor_map[i]:
            indices.append((i, j))
            values.append(
                weights[1] * get_norm(neighbor_map, j, inverse) / norm)
    print('@@@ normalize_adj_weighted_row_from_dict done')
    save(path, (indices, values, shape))
    return indices, values, shape


def normalize_batch_labels_weighted_row_from_dict(neighbor_map, batch,
                                                  weights=[0, 1, 0],
                                                  inverse=False):
    print('@@@ normalize_batch_labels_weighted_row_from_dict')
    path = '{}/../data/save/{}_weighted_row_norm_{}_{}_batch_{}.pickle'.format(
        current_folder,
        FLAGS.dataset,
        str(weights), \
        inverse, len(batch))
    rtn = load(path)
    if rtn:
        return rtn
    M = len(batch)
    N = len(neighbor_map)
    mat = np.zeros((M, N))
    for i, real_id in enumerate(batch):
        norm = np.sum([get_norm(neighbor_map, j, inverse) for j in neighbor_map[
            real_id]])
        for j in neighbor_map[real_id]:
            mat[i][j] = weights[1] * get_norm(neighbor_map, j, inverse) / norm
    print('@@@ normalize_batch_labels_weighted_row_from_dict done')
    # save(path, mat)
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


def construct_feed_dict(adj, features, support, labels, labels_mask,
                        placeholders, mode):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update(
        {placeholders['support'][i]: support[i] for i in range(len(support))})
    if mode == 0:
        if features is not None:
            feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['train_mask']: labels_mask})
        feed_dict.update({placeholders['ssl_labels']: labels})
    elif FLAGS.need_batch and mode == 2:
        # assert (type(labels) is dict)
        batch, pos_labels, neg_labels, labels = generate_batch(labels)
        # print('batch', batch)
        print('batch', batch.shape)
        # print('pos_labels', pos_labels)
        print('neg_labels', neg_labels)
        # print('labels', labels.shape)
        feed_dict.update({placeholders['batch']: batch})
        feed_dict.update({placeholders['pos_labels']: pos_labels})
        feed_dict.update({placeholders['neg_labels']: neg_labels})
        feed_dict.update({placeholders['usl_labels']: labels})
    elif mode == 2:
        feed_dict.update({placeholders['usl_labels']: labels})
        N = get_shape(labels)[0]
        inf_diagonal = np.zeros((N, N))
        np.fill_diagonal(inf_diagonal, 999999999999999999)
        feed_dict.update(
            {placeholders['sims_mask']: np.ones((N, N)) - np.identity(
                N) - inf_diagonal})
    return feed_dict


# data_index = 0
# round = 0
# batch_size = ceil(80513 // 4)
# ids = []
#
#
# def generate_batch(neighbor_map):
#     global data_index, round, ids
#     N = len(neighbor_map)
#     if round == 0:
#         ids = list(range(0, N))
#     end = data_index + batch_size
#     if end >= N:
#         end = N
#     batch = ids[data_index: end]
#     M = len(batch)
#     print('round %s \tbatch_size %s \t batch %s...' % (round, M, batch[0:5]))
#     rtn_labels = normalize_batch_labels_weighted_row_from_dict(neighbor_map,
#                                                                batch)
#     sims_mask = np.ones((M, N))
#     for i, j in enumerate(batch):
#         sims_mask[i][j] = -999999999999999999
#     data_index = end
#     if data_index == N:
#         data_index = 0
#         round += 1
#         shuffle(ids)
#     return batch, rtn_labels, sims_mask


data_index = 0
# max_size = 11799765 // 4
max_size = 667969
round = 0
ids = []
neg_sampler = NegSampler(num_neg=5)


def generate_batch(neighbor_map, num_neg=5):
    global data_index, round, ids
    if round == 0:
        ids = list(range(0, len(neighbor_map)))
        neg_sampler.init(len(neighbor_map))
    batch_size, num_data = get_size(neighbor_map, data_index, max_size)
    print('round: {} \tbatch_size: {} \t num_data: {}'.format(round,
                                                              batch_size,
                                                              num_data))
    batch = np.zeros(shape=(batch_size, 1))
    pos_labels = np.zeros(shape=(batch_size, 1))
    neg_labels = np.zeros(shape=(batch_size, num_neg))
    s = 0
    for i in range(num_data):
        id = get_id(neighbor_map, i + data_index)
        ns = neighbor_map[id]
        batch[s:s + len(ns), 0] = id
        pos_labels[s:s + len(ns), 0] = ns
        neg_labels[s:s + len(ns)] = neg_sampler.get_neg(neighbor_map[id])
        s += len(ns)
        neg_round = neg_sampler.increment()
        # print('@@@@@neg_round', neg_round)
    data_index = data_index + num_data
    if data_index >= len(neighbor_map):
        data_index = 0
        round += 1
        shuffle(ids)
    labels = np.concatenate(
        [np.ones((batch_size, 1)), np.zeros((batch_size, num_neg
                                             ))], 1)
    return batch, pos_labels, neg_labels, labels


def get_size(neighbor_map, data_index, max_size):
    size = 0
    i = get_id(neighbor_map, data_index)
    cnt = 0
    while size + len(neighbor_map[i]) <= max_size:
        size += len(neighbor_map[i])
        data_index += 1
        i = get_id(neighbor_map, data_index % len(neighbor_map))
        cnt += 1
    return size, cnt


def get_id(neighbor_map, idx):
    global ids
    return ids[idx % len(neighbor_map)]


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
