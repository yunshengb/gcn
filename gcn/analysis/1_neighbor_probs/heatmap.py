from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DIMENSION = 100
DIMENSIONS = [10, 100]


class Heatmap:
    def __init__(self, matrix):
        np.random.seed(0)
        sns.set()
        self.matrix = np.array(matrix)
        self.row = len(self.matrix)
        if self.row == 0:
            self.col = 0
        else:
            self.col = len(self.matrix[0])

    def getHeatmap(self, name, title, annot=False, i_s=None, i_e=None,
                   j_s=None, j_e=None):
        if not i_s:
            i_s = 0
        if not i_e:
            i_e = self.row - 1
        if not j_s:
            j_s = 0
        if not j_e:
            j_e = self.col - 1

        if i_s < 0 or i_e >= self.row:
            raise ValueError("No this row")
        if j_s < 0 or j_e >= self.col:
            raise ValueError("No this col")

        new_matrix = self.matrix[i_s:i_e + 1, j_s:j_e + 1]

        plt.figure("{}".format(name))
        sns.heatmap(new_matrix, cmap="YlGnBu", annot=annot)
        plt.title(title)
        plt.savefig("graph/{}.png".format(name))
        plt.close()


def predict_heatmap(filename):
    tokens = filename.split("_")
    tokens = tokens[1:]
    tokens[-1] = tokens[-1].split(".")[0]
    with open(filename) as f:
        i = 0
        for line in f:
            if i % 2 == 1:
                idx = i // 2
                round = tokens[idx]
                print('Round {} Probs matrix'.format(round))
                newline = line.strip("\n")
                new_mat = load_predict_matrix(newline)
                if round == '1000':
                    print
                for dim in DIMENSIONS:
                    new_heatmap = Heatmap(new_mat[0:dim, 0:dim])
                    name = "predict_round_{}_dim_{}".format(round, dim)
                    title = "Iteration {}".format(round)
                    new_heatmap.getHeatmap(name, title)
                    if idx == (len(tokens) - 1):
                        new_heatmap.getHeatmap(name + '_annot', title,
                                               annot=True)
            i += 1


def truth_heatmap(filename, norm=False):
    new_mat = load_truth_matrix(filename)
    print('Adjacency matrix')
    for dim in DIMENSIONS:
        new_heatmap = Heatmap(new_mat[0:dim, 0:dim])
        name = "truth_dim_{}".format(dim)
        if norm:
            name += '_norm'
        new_heatmap.getHeatmap(name, 'Adjacency Matrix')
        new_heatmap.getHeatmap(name + '_annot', 'Adjacency Matrix', True)

def mask_heatmap(filename, type):
    filename = type + filename
    mask = np.load(filename)
    print('Mask shape', mask.shape)
    for dim in DIMENSIONS:
        new_heatmap = Heatmap(mask[0:dim, 0:dim])
        name = '%s_mask_dim_%s' % (type, dim)
        new_heatmap.getHeatmap(name, 'Mask')
        new_heatmap.getHeatmap(name + '_annot', 'Mask', True)


def load_predict_matrix(newline):
    newline = newline.split("[")
    res_mat = []
    for line in newline:
        if line:
            numbers = line.strip("]").split(" ")
            res_mat.append(numbers)

    return np.array(res_mat, dtype=np.float32)


def load_truth_matrix(filename, dim=DIMENSION):
    with open(filename) as f:
        res_mat = []
        i = 0
        each_row = []
        for line in f:
            numbers = line.split(" ")

            if i == dim:
                i = 0
                res_mat.append(each_row)
                each_row = []
            elif i > dim:
                raise ValueError("More numbers in a row")

            for num in numbers:
                if num:
                    each_row.append(
                        num.strip("[").strip("]").strip("\n").strip("]"))
                    i += 1

    res_mat.append(each_row)
    return np.array(res_mat, dtype=int)


if __name__ == "__main__":
    # uniform_data = np.random.rand(100, 100)
    # heatmap = Heatmap(uniform_data)
    # heatmap.getHeatmap(0,99,0,99, "test")

    os.system('rm -rf graph/ && mkdir graph/')
    predict_heatmap(
        "probs_0_10_20_30_31_32_33_34_35_36_37_38_39_40_50_60_70_80_90_100_200_400_600_800_1000.txt")
    truth_heatmap("adj.txt")
    # truth_heatmap("adj_norm.txt", True)
    # mask_heatmap('_mask.mat', 'orig')