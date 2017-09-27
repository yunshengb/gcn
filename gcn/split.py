import numpy as np
import scipy.sparse as sp
import collections, os
from utils import load_data

LIMIT = 10000
DATASET = "flickr"

c = os.path.dirname(os.path.realpath(__file__))


def main():
    split_data = SplitData()
    split_data.gen_split()


class SplitData:
    def __init__(self):
        self.core_nodes = set()
        if DATASET == "cora":
            adj = read_cora()
            self.neigh_map = self._gen_neigh_map(adj)
            self.log = None
        elif DATASET == "flickr":
            self.neigh_map = read_flickr()
            self.log = open('log_split_flickr.txt', 'w')
        self.folder_id = 0
        self.total_nodes = 0
        os.system('rm -rf data/{}'.format(DATASET))

    def _gen_neigh_map(self, adj):
        print('Gen neighbor map: Getting edges')
        rows, cols = np.where(adj == 1)
        dic = collections.defaultdict(list)
        print('Gen neighbor map: Iterating')
        for idx, row in enumerate(rows):
            dic[row].append(cols[idx])
        print('Generated neighbor map')
        return dic

    def gen_split(self):
        for i in range(len(self.neigh_map)):
            if self.log:
                self.log.write('@@@@@ {}\n'.format(i))
            if i not in self.core_nodes:
                if self.log:
                    self.log.write('##### {}\n'.format(len(self.core_nodes)))
                self._BFS(i)

    def _BFS(self, i):
        fake_to_real = collections.defaultdict(int)
        real_to_fake = collections.defaultdict(int)
        real_cnt = 0
        q = collections.deque()
        q.append(i)
        local_visited = set()
        local_visited.add(i)
        while q:
            cur = q.popleft()
            if not self._check_neighbor(cur):
                if cur != i:
                    continue
                else:
                    self._drop_neighbor(cur)
            if self._need_add_to_core_area(fake_to_real, cur):
                # Add cur to the core area.
                self._add_to_id_map(real_to_fake, fake_to_real, cur)
                real_cnt += 1
                self.core_nodes.add(cur)
                for j in self.neigh_map[cur]:
                    if j not in local_visited:
                        local_visited.add(j)
                        # Add cur's neighbors to the extended area.
                        self._add_to_id_map(real_to_fake, fake_to_real, j)
                        q.append(j)
            else:
                # Matrix full.
                break
        self._dump_to_disk(fake_to_real, real_to_fake, real_cnt)

    def _add_to_id_map(self, real_to_fake, fake_to_real, id):
        if id in real_to_fake:
            return
        real_to_fake[id] = len(real_to_fake)
        fake_to_real[len(fake_to_real)] = id

    def _check_neighbor(self, id):
        # check if i's neighor is less than LIMIT, otherwise randomly\
        # drop some of i's neighbors
        if self._get_degree(id) > LIMIT:
            raise RuntimeError("For node#{}, Degree:{} > LIMIT".format(id,
                                                                       self._get_degree(
                                                                           id)))
        else:
            return True

    def _get_degree(self, id):
        return len(self.neigh_map[id])

    def _drop_neighbor(self, id):
        raise NotImplementedError()

    def _need_add_to_core_area(self, fake_to_real, id):
        # Two nodes may share the same neighbors, so the following is not
        # stringent.
        return len(fake_to_real) + self._get_degree(id) + 1 <= LIMIT and \
               id not in self.core_nodes

    def _dump_to_disk(self, fake_to_real, real_to_fake, real_cnt):
        path = "data/{}/{}".format(DATASET, self.folder_id)
        os.system("mkdir -p {}".format(path))
        self.total_nodes += real_cnt
        assert(len(self.core_nodes) == self.total_nodes)
        s = 'Folder {}, {} nodes / {}; total: {}\n'.format(self.folder_id,
                                                           real_cnt,
                                                           len(fake_to_real),
                                                           self.total_nodes)
        print(s)
        if self.log:
            self.log.write(s)
            self.log.write('{}\n'.format([fake_to_real[i] for i in range(
                real_cnt)]))
        self.folder_id += 1
        new_adj = np.zeros((len(fake_to_real), len(fake_to_real)))
        for real, fake in real_to_fake.items():
            for real_neigh in self.neigh_map[real]:
                if real_neigh in real_to_fake:
                    new_adj[fake][real_to_fake[real_neigh]] = 1
        sp.save_npz("{}/adj.npz".format(path), sp.csc_matrix(new_adj))
        np.save("{}/dim.npy".format(path), real_cnt)
        np.save("{}/map.npy".format(path), fake_to_real)
        # print('Nodes: {}\n'.format(real_to_fake.keys()))
        if self.total_nodes > len(self.neigh_map):
            raise RuntimeError('self.total_nodes {} > len(self.neigh_map) {}'
                               ''.format(self.total_nodes, len(self.neigh_map)))


def read_cora():
    adj, _, _, _, _, _, _, _ = load_data("cora", 0)
    return adj.todense()


def read_flickr():
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
    return dic


if __name__ == "__main__":
    main()
