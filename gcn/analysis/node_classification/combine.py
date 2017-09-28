import os, glob
import numpy as np

c_folder = os.path.dirname(os.path.realpath(__file__))
DIM = (0,0)

class Data_combiner:
    def __init__(self, dataset, model):
        self.data = self.combiner(dataset, model)
        self.save_data()
        self.dim = DIM

    def save_data(self):
        pass

    def load_data(self):
        pass

    def get_data(self):
        return self.data

    def set_dim(self, dim):
        self.dim = dim

    def combiner(self, dataset, model):
        path = "{}/{}/{}".format(c_folder, dataset, model)
        combined_data = np.zeros(self.dim)
        for folder in os.listdir(path):
            emb_file = path + "/{}/emb.npy".format(folder)
            map_file = path + "/{}/map.npy".format(folder)
            emb = np.load(emb_file)
            map = np.load(map_file)
            for fake_node in map:
                combined_data[map[fake_node]] = emb[fake_node]
        return combined_data
