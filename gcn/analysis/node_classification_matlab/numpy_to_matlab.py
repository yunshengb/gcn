import numpy as np
import scipy.io, os

c = os.path.dirname(os.path.realpath(__file__))

# NP_MAT = '../../../../node2vec/emb/node2vec_blog_emb.npy'
# MATLAB_MAT = 'node2vec_blog_emb.mat'

# NP_MAT = '../../data/BlogCatalog-dataset/data/blog_labels.npy'
# MATLAB_MAT = 'blog_labels.mat'

NP_MAT = 'gcn_blog_emb_1000.npy'
MATLAB_MAT = 'gcn_blog_emb_1000.mat'

def main():
    NP_MAT_ = '{}/{}'.format(c, NP_MAT)
    MATLAB_MAT_ = '{}/{}'.format(c, MATLAB_MAT)
    print('Loading {}'.format(NP_MAT_))
    arr = np.load(NP_MAT)
    print('Dumping {}'.format(MATLAB_MAT_))
    scipy.io.savemat(MATLAB_MAT_, mdict={'arr': arr})


if __name__ == '__main__':
    main()
