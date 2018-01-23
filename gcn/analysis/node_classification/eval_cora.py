import os, glob
import numpy as np

c_folder = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, os.path.join(c_folder, "../.."))
# sys.path.insert(0, os.path.join(c_folder, "../../../../liblinear/python"))
# from liblinearutil import *

GCN_EXP = \
    "gcn_cora_embed_nogrow_2nd_20180122105653"
main_file = '{}/../../src/train.py'.format(c_folder)




embeddings = glob.glob('{}/../../exp/{}/gcn_cora_emb_*.npy'.format(c_folder,
                                                                   GCN_EXP))
print(embeddings)
for emb in sorted(embeddings):
    if 'result' not in emb:
        continue
        emb = GCN_EXP + '/' + emb.split('/')[-1].split('.')[0]
        print(emb)
        os.system('python {} --eval {} --epochs 500 --debug 1 --embed 0'.format(
            main_file, emb))
        check_file = '{}/../exp/{}_result'.format(c_folder, emb)
        if os.path.isfile(check_file):
            print('Error: should have {}'.format(check_file))
            exit(1)
    else:
        result = np.load(emb)
        print(result)

