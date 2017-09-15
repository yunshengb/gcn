from __future__ import print_function
from heatmap import Heatmap
import numpy as np
import tensorflow as tf
import os, glob, re

c = os.path.dirname(os.path.realpath(__file__))

datasets = ['blog']

DIMENSIONS = [10, 100]

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def main():
    os.system('rm -rf graph/ && mkdir graph/')
    models = [Model('node2vec', '%s/../../../../node2vec/emb' % c),
              Model('gcn', get_latest_gcn_exp())]
    for dataset in datasets:
        for dim in DIMENSIONS:
            truth = np.load('%s_adj_norm.npy' % dataset)
            Heatmap(truth[0:dim, 0:dim]).getHeatmap(
                'truth_sims_dim_%s' % dim, 'Adjacency Normalized')
        for model in models:
            model.eval(dataset, truth)


def get_latest_gcn_exp():
    dirs = glob.glob('%s/../../gcn_*' % c)
    max_ts = -np.inf
    rtn = ''
    for dir in dirs:
        if os.path.isdir(dir):
            ts = int(dir.split('/')[-1].split('_')[-1])
            if ts > max_ts:
                max_ts = ts
                rtn = dir
    return '%s/intermediate' % rtn


class Model(object):
    def __init__(self, name, dir):
        self.name = name
        self.dir = dir
        print('%s data in %s' % (name, dir))

    def eval(self, dataset, truth):
        files = sort_nicely(glob.glob('%s/%s_%s_emb*.npy' % (self.dir,
                                                             self.name,
                                                             dataset)))
        for file in files:
            self._eval(file, dataset, truth)

    def _eval(self, file, dataset, truth):
        epoch_str = self._get_epoch_str(file)
        emb = np.load(file)
        sims = self._get_sims(emb, truth)
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

    def _get_epoch_str(self, file):
        try:
            epoch = int(file.split('.')[-2].split('_')[-1])
            epoch_str = '_epoch_%s' % epoch
        except ValueError:
            epoch_str = ''
        return epoch_str

    def _get_sims(self, emb, truth):
        sims = np.dot(emb, emb.T)
        inf_diagonal = np.zeros(truth.shape)
        np.fill_diagonal(inf_diagonal, 999999999999999999)
        sims = np.multiply(sims, (np.ones(sims.shape) - np.identity(
            sims.shape[0]) - inf_diagonal))
        if sims.shape != truth.shape:
            raise RuntimeError('emb %s truth %s' % (sims.shape, truth.shape))
        return sims

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


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """

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

    l.sort(key=alphanum_key)
    return l


if __name__ == '__main__':
    main()
