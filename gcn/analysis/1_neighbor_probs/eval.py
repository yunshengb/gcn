from heatmap import Heatmap
import numpy as np
import tensorflow as tf
import os

c = os.path.dirname(os.path.realpath(__file__))

datasets = ['cora']

DIM = 50

sess = tf.Session()
sess.run(tf.global_variables_initializer())


class Model(object):
    def __init__(self, name, dir):
        self.name = name
        self.dir = dir

    def eval(self, dataset, truth):
        emb = np.load('%s/%s_emb.npy' % (self.dir, dataset))
        sims = np.dot(emb, emb.T)
        sims = np.multiply(sims, (np.ones(sims.shape) - np.identity(
            sims.shape[0])))
        if sims.shape != truth.shape:
            raise RuntimeError('emb %s truth %s' % (sims.shape, truth.shape))
        sims_, truth_, probs, loss = self._loss(sims, truth)
        loss, probs = sess.run([loss, probs], feed_dict={sims_: sims, truth_: truth})
        print(self.name, 'loss', loss)
        Heatmap(probs[0:DIM, 0:DIM]).getHeatmap(
            '%s_%s_sims' % (self.name, dataset), '%s %s embeddings'
            % (self.name, dataset))
        print(probs[0:4, 0:4])

    def _loss(self, sims, truth):
        sims_ = tf.placeholder(tf.float32, shape=sims.shape)
        truth_ = tf.placeholder(tf.float32, shape=truth.shape)
        return sims_, truth_, tf.nn.softmax(sims_), tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=sims_, labels=truth_))


def main():
    models = [Model('node2vec', '%s/../../../../node2vec/emb' % c),
              Model('gcn', '%s/../../intermediate' % c)]
    truth = np.load('adj_norm.npy')
    Heatmap(truth[0:DIM, 0:DIM]).getHeatmap(
        'truth_sims', 'Adjacency Normalized')
    for dataset in datasets:
        for model in models:
            model.eval(dataset, truth)


if __name__ == '__main__':
    main()
