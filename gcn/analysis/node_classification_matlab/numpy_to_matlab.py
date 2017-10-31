import numpy as np
import scipy.io, os, glob

c = os.path.dirname(os.path.realpath(__file__))

# NP_MAT = '../../../../node2vec/emb/node2vec_blog_emb.npy'
# MATLAB_MAT = 'node2vec_blog_emb.mat'

# NP_MAT = '../../data/BlogCatalog-dataset/data/blog_labels.npy'
# MATLAB_MAT = 'blog_labels.mat'

NP_MAT = 'blog_labels.npy'
# MATLAB_MAT = 'blog_emb_pq_grid_search/*.mat'

# NP_MAT = '../../data/arxiv/arxiv_adj.npy'
# MATLAB_MAT = '../../data/arxiv/arxiv_adj.mat'

def main():
    for file in glob.glob(NP_MAT):
        NP_MAT_ = '{}/{}'.format(c, file)
        MATLAB_MAT = file.replace('npy', 'mat')
        MATLAB_MAT_ = '{}/{}'.format(c, MATLAB_MAT)
        print('Loading {}'.format(NP_MAT_))
        arr = np.load(NP_MAT_)
        print('Dumping {}'.format(MATLAB_MAT_))
        scipy.io.savemat(MATLAB_MAT_, mdict={'arr': arr})


if __name__ == '__main__':
    main()
