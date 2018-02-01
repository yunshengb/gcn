import os, glob
import numpy as np

c_folder = os.path.dirname(os.path.realpath(__file__))
DS = 'blog'

GCN_EXP = \
    "gcn_blog_embed_2nd_5_2_2_1_20171219010711"
main_file = '{}/../../src/train.py'.format(c_folder)
TR = 0.8



embeddings = glob.glob('{}/../../exp/{}/gcn_*_emb_*.npy'.format(c_folder,
                                                                   GCN_EXP))
print(embeddings)
for emb in sorted(embeddings):
    if 'result' not in emb:
        #continue
        emb = GCN_EXP + '/' + emb.split('/')[-1].split('.')[0]
        print(emb)
        os.system('python {} --dataset {} --eval {} --epochs 300 --debug 1 '
                  '--embed 0 '
                  '--train_ratio {} '
                  '--device cpu'
                  ''.format(
            main_file, DS, emb, TR))
        check_file = '{}/../exp/{}_result'.format(c_folder, emb)
        if os.path.isfile(check_file):
            print('Error: should have {}'.format(check_file))
            exit(1)
    else:
        result = np.load(emb)
        print(result)

