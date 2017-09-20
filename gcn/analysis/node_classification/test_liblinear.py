import sys,os
import numpy as np
from sklearn.model_selection import train_test_split
c_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(c_folder, "../../../../liblinear/python"))

from liblinearutil import *

emb = np.load("blog/node2vec/blog_emb_iter_1_p_1_q_1_walk_10_win_1.npy")
label = np.load("blog/data/blog_labels.npy")

X_train, X_test, y_train, y_test = train_test_split(emb, label, train_size=0.5)

prob = problem(y_train, X_train)

for i in range(8):
    param = parameter('-s {} -c 4 -B 1'.format(str(i)))
    m = train(prob, param)
    p_label, p_acc, p_val = predict(y_test, X_test, m, '-b 1')
    ACC, MSE, SCC = evaluations(y_test, p_label)