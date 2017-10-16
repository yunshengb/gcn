import numpy as np

NUM_NODES = 5242

id_map = {}


def main():
    adj()


def adj():
    adj = np.zeros((NUM_NODES, NUM_NODES))
    num_edges = 0
    with open('CA-GrQc.txt') as f:
        for i in range(4):
            next(f) # ignore 4 lines of header
        for line in f:
            ls = line.rstrip().split('\t')
            x = id(ls[0])
            y = id(ls[1])
            num_edges = connect(adj, x, y, num_edges)
            num_edges = connect(adj, y, x, num_edges)
    np.save('arxiv_adj.npy', adj)
    print('arxiv_adj.npy saved with shape {} and {} edges'.format(adj.shape, num_edges))


def connect(adj, i, j, num_edges):
    if adj[i][j] != 1:
        adj[i][j] = 1
        num_edges += 1
    return num_edges

def id(i):
    i = int(i)
    if not i in id_map:
        id_map[i] = len(id_map)
    return id_map[i]


if __name__ == '__main__':
    main()
