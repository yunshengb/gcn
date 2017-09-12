import tensorflow as tf
from gcn.utils import *
from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import glob,os,re,sys,collections,pickle
import numpy as np
import matplotlib.pyplot as plt

c_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(c_folder, "../.."))

LIM = None
class Data_engine:
    def __init__(self,dataset):
        self.dataset = dataset
        self.eval = {}
        self.loss = {}
        self.run_loss = None
        self.run_eval = None
        self.baseline = (0,0)

        if dataset == "blog":
            self.run_eval = run_blog
        elif dataset == "cora":
            self.run_eval = run_cora
            self.run_loss = run_cora_loss
    def run(self,model):
        self.eval = {**self.eval, **self.run_eval(model)}
        self.save_eval()
    def run_with_loss(self,model):
        self.loss = {**self.loss, **self.run_loss(model)}
    def load_eval(self):
        with open('eval.pkl', 'rb') as f:
            return pickle.load(f)
    def save_eval(self):
        with open('eval.pkl', 'wb') as f:
            pickle.dump(self.eval, f, pickle.HIGHEST_PROTOCOL)
    def pre_draw(self):
        data = collections.defaultdict(list)
        for folder, file in self.eval:
            if 'gcn' in folder:
                file_name = file.split("/")[-1]
                x = int(file_name.split('.')[0].split("_")[-1])
                data[folder].append((x, self.eval[(folder, file)]))
            if 'node2vec' in folder:
                self.baseline = self.eval[(folder, file)]
        return data
    def pre_draw_loss(self):
        data = collections.defaultdict(list)
        for folder, file in self.loss:
            if 'gcn' in folder:
                file_name = file.split("/")[-1]
                x = int(file_name.split('.')[0].split("_")[-1])
                data[folder].append((x, self.loss[(folder, file)]))
        return data
    def draw(self):
        eval = self.pre_draw()
        loss = self.pre_draw_loss()

        for folder in eval:
            accs = []
            f1s = []
            xs = []

            for x, (acc,f1) in eval[folder]:
                accs.append(acc)
                f1s.append(f1)
                xs.append(x)

            losses = []
            l_xs = []

            base_ys = [self.baseline[0]] * len(xs)

            for x, l in loss[folder]:
                l_xs.append(x)
                losses.append(l)

            fig, ax = plt.subplots()
            fig.suptitle(folder)

            def two_scales(ax1, xs, l_xs, base_ys, accs, losses, c1, c2):
                ax2 = ax1.twinx()
                ax1.plot(xs, accs, color=c1, label = "gcn acc")
                ax1.plot(xs, base_ys, color = 'yellowgreen', label = 'node2vec acc', linestyle = "--")
                ax1.legend(loc="upper right")
                ax1.set_xlabel('Iter')
                # ax1.set_xlim([0, 100])
                ax1.set_ylabel('Accs')
                ax1.yaxis.label.set_color(c1)

                ax2.plot(l_xs, losses, color=c2, label = "loss")
                # ax2.set_xlim([0, 100])
                ax2.set_ylabel('Losses')
                ax2.yaxis.label.set_color(c2)

                def color_y_axis(ax, color):
                    """Color your axes."""
                    for t in ax.get_yticklabels():
                        t.set_color(color)
                    return None

                color_y_axis(ax1, c1)
                color_y_axis(ax2, c2)


                return ax1, ax2

            ax1, ax2 = two_scales(ax, xs, l_xs, base_ys, accs, losses, 'lightcoral', 'lightskyblue')
            plt.savefig(folder + ".png")
            # plt.show()

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

def run_cora(model):
    print('Load Data')
    eval = collections.defaultdict(list)
    adj, features, y_labels, y_val, y_truth, train_mask, val_mask, test_mask = load_data('cora', 0)
    dataset = (adj, features, y_labels, y_val, y_truth, train_mask, val_mask, test_mask)
    if model == "gcn":
        for folder in sort_nicely(os.listdir(os.path.join(c_folder, "cora/gcn"))):
            path = c_folder + "/cora/gcn/{}/intermediate".format(folder)
            for file in sort_nicely(glob.glob(path + "/*.npy")):
                if "emb" in file and "loss" not in file:
                    print("*" * 50)
                    print('Processing folder ', folder)
                    print ('Processing file ', file.split("/")[-1])
                    acc, f1 = run_one_file_cora(file, dataset)
                    eval[(folder,file)] = [acc,f1]

    elif model == "node2vec":
        path = c_folder + "/cora/node2vec"
        for file in sort_nicely(glob.glob(path + "/*.npy")):
            print("*" * 50)
            print('Processing file ', file.split("/")[-1])
            acc, f1 = run_one_file_cora(file, dataset)
            eval[("node2vec",file)] = [acc,f1]
    return eval
def run_cora_loss(model):
    print('Load Loss')
    loss = collections.defaultdict(float)
    if model == "gcn":
        for folder in sort_nicely(os.listdir(os.path.join(c_folder, "cora/gcn"))):
            path = c_folder + "/cora/gcn/{}/intermediate".format(folder)
            for file in sort_nicely(glob.glob(path + "/*.npy")):
                if "loss" in file and "emb" not in file:
                    print("*" * 50)
                    print('Processing folder ', folder)
                    print ('Processing file ', file.split("/")[-1])
                    cur_loss = np.load(file)
                    print('loss = ', cur_loss)
                    loss[(folder,file)] = float(cur_loss)
    return loss
def run_one_file_cora(file, dataset):
    embed = np.load(file)
    adj, features, y_labels, y_val, y_truth, train_mask, val_mask, test_mask = dataset
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
# def gen_eval_input(eval, acc, f1, embedding):
#     embedding = embedding.split("/")[-1]
#     name_tokens = embedding[:-4].split("_")
#     # key: (iter, p, q, num_walks, window_size)
#     key = []
#     for idx, token in enumerate(name_tokens):
#         if token in ["iter", "p", "q", "walk", "win"]:
#             key.append(name_tokens[idx + 1])
#     assert (len(key) == 5)
#     eval[tuple(key)] = [acc, f1]

if __name__ == '__main__':
    data_engine = Data_engine("cora")
    data_engine.run("gcn")
    data_engine.run_with_loss('gcn')
    data_engine.run("node2vec")
    data_engine.draw()