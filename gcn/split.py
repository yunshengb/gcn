import numpy as np
import collections, random, os
from utils import load_data

LIMIT = 20000
DATASET = "cora"

def main():
    split_data = SplitData(DATASET)
    split_data.gen_split()

class SplitData:
    def __init__(self, dataset):
        self.visited = set()
        self.adj = None
        if dataset == "cora":
            self.adj = read_cora()

        self.neigh_map = self._gen_neigh_map()
        self.folder_id = 0
        self.total_nodes = 0

    def _gen_neigh_map(self):
        rows, cols = np.where(self.adj == 1)
        dic = collections.defaultdict(list)
        for idx, row in enumerate(rows):
            dic[row].append(cols[idx])
        return dic

    def gen_split(self):
        for i in range(self.adj.shape[0]):
            if i not in self.visited:
                self.visited.add(i)
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
                break
            else:
                # Add cur to the core area.
                self._add_to_id_map(real_to_fake, fake_to_real, cur)
                real_cnt += 1
                self.visited.add(cur)
                for j in self.neigh_map[i]:
                    if j not in local_visited:
                        local_visited.add(j)
                        # Add cur's neighbors to the extended area.
                        self._add_to_id_map(real_to_fake, fake_to_real, j)
                        q.append(j)

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
            raise RuntimeError("For node#{}, Degree:{} > LIMIT".format(id, self._get_degree(id)))
        else:
            return True

    def _get_degree(self, id):
        return len(self.neigh_map[id])

    def _drop_neighbor(self, id):
        raise NotImplementedError()

    def _need_add_to_core_area(self, fake_to_real, id):
        # Two nodes may share the same neighbors, so the following is not\
        # stringent
        return len(fake_to_real) + self._get_degree(id) + 1 > LIMIT

    def _dump_to_disk(self, fake_to_real, real_to_fake, real_cnt):
        path = "data/{}/{}".format(DATASET,self.folder_id)
        os.system("mkdir -pv {}".format(path))
        print("Create Folder #{}".format(self.folder_id))
        self.folder_id += 1
        new_adj = np.zeros((len(fake_to_real),len(fake_to_real)))
        for real, fake in real_to_fake.items():
            for real_neigh in self.neigh_map[real]:
                if real_neigh in real_to_fake:
                    new_adj[fake][real_to_fake[real_neigh]] = 1
        np.save("{}/adj.npy".format(path),new_adj)
        np.save("{}/dim.npy".format(path),real_cnt)
        np.save("{}/map.npy".format(path), fake_to_real)
        self.total_nodes += real_cnt
        print (self.total_nodes)


def read_cora():
    adj, _, _, _, _, _, _, _ = load_data("cora", 0)
    return adj.todense()

if __name__ == "__main__":
    main()