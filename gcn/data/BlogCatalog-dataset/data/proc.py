import numpy as np

NUM_NODES = 10312
NUM_GROUP = 39


def main():
    adj()
    labels()


def adj():
    adj = np.zeros((NUM_NODES, NUM_NODES))
    with open('edges.csv') as f:
        for line in f:
            ls = line.rstrip().split(',')
            x = id(ls[0])
            y = id(ls[1])
            adj[x][y] = 1
            adj[y][x] = 1
    np.save('blog_adj.npy', adj)
    print('blog_adj.npy saved with shape {}'.format(adj.shape))


def labels():
    labels = np.zeros((NUM_NODES, NUM_GROUP))
    with open('group-edges.csv') as f:
        for line in f:
            ls = line.rstrip().split(',')
            node_id = id(ls[0])
            group_id = id(ls[1])
            labels[node_id][group_id] = 1
    np.save('blog_labels.npy', labels)
    print('blog_labels.npy saved with shape {}'.format(labels.shape))


def id(i):
    return int(i) - 1


if __name__ == '__main__':
    main()
