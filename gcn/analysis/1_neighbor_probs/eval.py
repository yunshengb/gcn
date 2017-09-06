from __future__ import print_function
from heatmap import Heatmap
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
import os, glob, re

c = os.path.dirname(os.path.realpath(__file__))

datasets = ['cora']

DIMENSIONS = [10, 100]

sess = tf.Session()
sess.run(tf.global_variables_initializer())


class Model(object):
    def __init__(self, name, dir):
        self.name = name
        self.dir = dir

    def eval(self, dataset, truth):
        files = sort_nicely(glob.glob('%s/%s_%s_emb*.npy' % (self.dir,
                                                             self.name,
                                                             dataset)))
        for file in files:
            self._eval(file, dataset, truth)

    def _eval(self, file, dataset, truth):
        try:
            epoch = int(file.split('.')[-2].split('_')[-1])
            epoch_str = '_epoch_%s' % epoch
        except ValueError:
            epoch_str = ''
        emb = np.load(file)
        # emb = normalize(emb, norm='l2')
        sims = np.dot(emb, emb.T)
        inf_diagonal = np.zeros(truth.shape)
        np.fill_diagonal(inf_diagonal, 999999999999999999)
        sims = np.multiply(sims, (np.ones(sims.shape) - np.identity(
            sims.shape[0]) - inf_diagonal))
        if sims.shape != truth.shape:
            raise RuntimeError('emb %s truth %s' % (sims.shape, truth.shape))
        sims_, truth_, probs, loss = self._loss(sims, truth)
        loss, probs = sess.run([loss, probs],
                               feed_dict={sims_: sims, truth_: truth})
        print('%s%s loss %s' % (self.name, epoch_str, loss))
        for dim in DIMENSIONS:
            Heatmap(probs[0:dim, 0:dim]).getHeatmap(
                '%s_%s_probs_dim_%s%s' % (self.name, dataset, dim, epoch_str),
                '%s %s embeddings'
                % (self.name, dataset))
            # print(probs[0:4, 0:4])

    def _loss(self, sims, truth):
        sims_ = tf.placeholder(tf.float32, shape=sims.shape)
        truth_ = tf.placeholder(tf.float32, shape=truth.shape)
        return sims_, truth_, tf.nn.softmax(sims_), tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=sims_, labels=truth_))


'''
Code below is from 
https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically.
'''


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l


def main():
    os.system('rm -rf graph/ && mkdir graph/')
    models = [Model('node2vec', '%s/../../../../node2vec/emb' % c),
              Model('gcn', '%s/../../intermediate' % c)]
    truth = np.load('adj_norm.npy')
    for dim in DIMENSIONS:
        Heatmap(truth[0:dim, 0:dim]).getHeatmap(
            'truth_sims_dim_%s' % dim, 'Adjacency Normalized')
    for dataset in datasets:
        for model in models:
            model.eval(dataset, truth)


if __name__ == '__main__':
    main()
