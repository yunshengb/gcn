import tensorflow as tf
from gcn.utils import *
from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import glob,os,re,sys,collections
import numpy as np
import matplotlib.pyplot as plt

c_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(c_folder, "../.."))

class Data_engine:
    def __init__(self,dataset):
        self.dataset = dataset
        self.eval = {}
        if dataset == "blog":
            self.run_func = run_blog
            self.pre_draw = pre_draw_blog
    def run(self,model):
        self.eval = {**self.eval, **self.run_func(model)}
    def draw(self):
        data = self.pre_draw(self.eval)
        for folder in data:
            accs = []
            f1s = []
            xs = []

            for x, (acc,f1) in data[folder]:
                accs.append(acc)
                f1s.append(f1)
                xs.append(x)

            plt.figure(folder)
            plt.plot(xs, f1s)
            plt.plot(xs, accs)
            plt.legend(['f1', 'acc'], loc='upper right')
            plt.savefig(folder + '.png')

            plt.show()

def pre_draw_blog(eval):
    data = collections.defaultdict(list)
    for folder,file in eval:
        if 'gcn' in folder:
            file_name = file.split("/")[-1]
            x = int(file_name.split('.')[0].split("_")[-1])
            data[folder].append((x,eval[(folder,file)]))
    return data
def run_blog(model):
    print("Load Data")
    labels = np.load(os.path.join(c_folder, "blog/data/blog_labels.npy"))
    eval = collections.defaultdict(list)
    if model == "gcn":
        for folder in sort_nicely(os.listdir(os.path.join(c_folder, "blog/gcn"))):
            path = c_folder + "/blog/gcn/{}/intermediate".format(folder)
            for file in sort_nicely(glob.glob(path + "/*.npy")):
                print("*" * 50)
                print('Processing folder ', folder)
                print ('Processing file ', file.split("/")[-1])
                embedding = np.load(file)
                acc, f1 = run_one_file(embedding, labels)
                eval[(folder,file)] = [acc,f1]

    elif model == "node2vec":
        path = c_folder + "/blog/node2vec"
        for file in sort_nicely(glob.glob(path + "/*.npy")):
            print("*" * 50)
            print('Processing file ', file.split("/")[-1])
            embedding = np.load(file)
            acc, f1 = run_one_file(embedding, labels)
            eval[("node2vec",file)] = [acc,f1]
    return eval
def run_one_file(embedding, labels):
    X_train, X_test, y_train, y_test = train_test_split(embedding, labels)
    acc, f1 = run_model_sklearn(X_train, y_train, X_test, y_test)
    return acc, f1

def run_model_sklearn(X_train, y_train, X_test, y_test):
    if len(y_train[0]) > 1:
        y_train = process_y(y_train)
    if len(y_test[0]) > 1:
        y_test = process_y(y_test)
    lr = LogisticRegression(multi_class= 'ovr', solver = 'liblinear')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print('Result')
    acc = accuracy_score(y_test, y_pred)
    print ('acc = ', acc)
    f1 = f1_score(y_test, y_pred, average='macro')
    print('f1_score = ', f1)
    return acc, f1
def process_y(y_data):
    y_res = []
    for entry in y_data:
        if max(entry) == 0.0:
            print(entry)
            y_res.append(len(entry))
        else:
            index = list(entry).index(max(entry))
            y_res.append(index)
    y_res = np.asarray(y_res)
    return y_res
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

def run_dataset(dataset):
    print('Python 3 please!')
    print('Load embeddings')
    for embedding in sort_nicely(glob.glob(os.path.join(c_folder + "/npy_files", "*.npy"))):
        acc, f1 = analyze_embedding(embedding, dataset)
    return
def gen_eval_input(eval, acc, f1, embedding):
    embedding = embedding.split("/")[-1]
    name_tokens = embedding[:-4].split("_")
    # key: (iter, p, q, num_walks, window_size)
    key = []
    for idx, token in enumerate(name_tokens):
        if token in ["iter", "p", "q", "walk", "win"]:
            key.append(name_tokens[idx + 1])

    assert (len(key) == 5)
    eval[tuple(key)] = [acc, f1]
def analyze_embedding(embedding, dataset):
    embed = np.load(embedding)
    print ("*" * 50)
    print('Processing file ', embedding.split("/")[-1])
    adj, features, y_labels, y_val, y_truth, train_mask, val_mask, test_mask = load_data(dataset, 0)
    X_train, y_train, X_test, y_test = split_data(embed, features, y_labels, y_val,y_truth, train_mask, val_mask, test_mask)
    acc, f1 = run_model_sklearn(X_train, y_train, X_test, y_test)
    return acc, f1
def split_data(embed, features, y_labels, y_val,y_truth, train_mask, val_mask, test_mask):
    data = np.concatenate((embed, features.toarray()), axis=1)
    X_train = []
    y_train = []
    for i in range(len(train_mask)):
        if train_mask[i]:
            X_train.append(data[i])
            y_train.append(y_labels[i])

    for i in range(len(val_mask)):
        if val_mask[i]:
            X_train.append(data[i])
            y_train.append(y_val[i])
    X_test = []
    y_test = []

    for i in range(len(test_mask)):
        if test_mask[i]:
            X_test.append(data[i])
            y_test.append(y_truth[i])

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    return X_train, y_train, X_test, y_test

'''
def run_model(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(input=X_train, label=y_train, n_in=X_train.shape[1], n_out=y_train.shape[1])
    n_epochs = 200
    learning_rate = 0.01
    for epoch in range(n_epochs):
        lr.train(lr=learning_rate, L2_reg = 1.0)
        # cost = lr.negative_log_likelihood()
        # if epoch % 20 == 0:
        #     print('Training epoch %d, cost is %f' % (epoch, cost))
        learning_rate *= 0.95

    print('Result')
    y_pred = lr.predict(X_test)
    acc = cal_accuracy(y_pred, y_test)
    f1 = cal_macro_F1(y_pred, y_test)

    return acc, f1

def cal_accuracy(y_pred, y_test):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    sess = tf.Session()
    acc_result = sess.run(accuracy_all)
    acc = sum(acc_result)/len(acc_result)
    print('acc = ',acc)
    return acc

def cal_macro_F1(y_pred, y_test):
    y_pred_res = []
    for entry in y_pred:
        new_entry = [0] * len(entry)
        index = list(entry).index(max(entry))
        new_entry[index] = 1
        y_pred_res.append(new_entry)
    y_pred_res = np.asarray(y_pred_res)

    f1 = f1_score(y_pred_res, y_test, average = 'macro')
    print ('f1_score = ', f1)
    return f1
'''
'''
Code below is from 
https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically.
'''

if __name__ == '__main__':
    data_engine = Data_engine("blog")
    data_engine.run("node2vec")
    # data_engine.run("node2vec")
    #data_engine.draw()