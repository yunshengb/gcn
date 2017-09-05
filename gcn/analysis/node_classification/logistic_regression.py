import numpy as np


def main():
    print('Python 3 please!')

    print('Load embeddings')

    print(np.load('gcn_cora_emb.npy').shape)
    print(np.load('node2vec_cora_emb.npy').shape)

    print('Load ground-truths labels for cora')

    print('Load features and append to embeddings')

    print('Perform logistic regression')

    print('Result')


if __name__ == '__main__':
    main()