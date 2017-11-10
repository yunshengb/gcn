from gcn.utils import *
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import glob, os, re, sys, collections, pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance
from sklearn.naive_bayes import GaussianNB

c_folder = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, os.path.join(c_folder, "../.."))
# sys.path.insert(0, os.path.join(c_folder, "../../../../liblinear/python"))
# from liblinearutil import *

GCN_EXP = "gcn_flickr_weighted_adj_alpha_0_7_beta_0_3_forward_20171109114215"
GCN_FOLDER = "{}/../../exp/{}/intermediate".format(c_folder, GCN_EXP)
NODE2VEC_FOLDER = "{}/../../../../node2vec/emb".format(c_folder)


class Data_engine:
    def __init__(self, dataset):
        self.dataset = dataset
        self.eval = {}
        self.loss = {}
        self.run_loss = None
        self.run_eval = None
        self.baseline = (0, 0)
        if dataset == "blog":
            self.run_eval = run_blog
            self.run_loss = run_blog_loss
        elif dataset == "cora":
            self.run_eval = run_cora
            self.run_loss = run_cora_loss
        elif dataset == "flickr":
            self.run_eval = run_flickr

    def run(self, model, folder=None):
        self.eval = {**self.eval, **self.run_eval(model, folder)}
        self.save_eval()

    def run_with_loss(self, model, folder=None):
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

    def two_scales(self, ax1, xs, l_xs, base_measures, measures, losses, c1, c2,
                   measure):
        ax2 = ax1.twinx()
        ax1.plot(xs, measures, color=c1, label="gcn {}".format(measure))
        ax1.plot(xs, base_measures, color='yellowgreen',
                 label='node2vec {}'.format(measure), linestyle="--")
        ax1.legend(loc="upper right")
        ax1.set_xlabel('Iter')
        # ax1.set_xlim([0, 100])
        ax1.set_ylabel('{}'.format(measure))
        ax1.yaxis.label.set_color(c1)
        ax2.plot(l_xs, losses, color=c2, label="loss")
        # ax2.set_xlim([0, 100])
        ax2.set_ylabel('Losses')
        ax2.yaxis.label.set_color(c2)
        self.color_y_axis(ax1, c1)
        self.color_y_axis(ax2, c2)
        return ax1, ax2

    def color_y_axis(self, ax, color):
        """Color your axes."""
        for t in ax.get_yticklabels():
            t.set_color(color)
        return None

    def draw(self, measure):
        eval = self.pre_draw()
        loss = self.pre_draw_loss()

        for folder in eval:
            accs = []
            f1s = []
            xs = []
            for x, (acc, f1) in eval[folder]:
                accs.append(acc)
                f1s.append(f1)
                xs.append(x)

            losses = []
            l_xs = []

            base_acc = [self.baseline[0]] * len(xs)
            base_f1 = [self.baseline[1]] * len(xs)

            for x, l in loss[folder]:
                l_xs.append(x)
                losses.append(l)

            fig, ax = plt.subplots()
            fig.suptitle(folder + "_" + measure)
            if measure == "f1":
                self.two_scales(ax, xs, l_xs, base_f1, f1s, losses,
                                'lightcoral', 'lightskyblue', measure)
            elif measure == "acc":
                self.two_scales(ax, xs, l_xs, base_acc, accs, losses,
                                'lightcoral', 'lightskyblue', measure)
            plt.savefig(folder + "_" + measure + ".png")


def run_flickr(model, folder=None):
    eval = collections.defaultdict(list)
    labels = np.load(c_folder + "/flickr/data/flickr_labels.npy")
    if model == "node2vec":
        path = c_folder + "/flickr/node2vec"
        for file in sort_nicely(glob.glob(path + "/*.npy")):
            print('*' * 50)
            print('Processing model {}'.format(model))
            embedding = np.load(file)
            acc, f1 = run_one_file_flickr(embedding, labels)
            eval[(model, "test")] = [acc, f1]
    return eval


# def run_one_file_flickr(embedding, labels):
#     X_train, X_test, y_train, y_test = train_test_split(embedding, labels, train_size=0.5)
#     ACC, f1 = run_model_sklearn(X_train, y_train, X_test, y_test)
#     return ACC, f1

def run_one_file_flickr(embedding, labels):
    X_train, X_test, y_train, y_test = train_test_split(embedding, labels,
                                                        train_size=0.5)
    classif = OneVsRestClassifier(LogisticRegression(class_weight="balanced"))
    classif.fit(X_train, y_train)
    y_pred = classif.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    ACC = accuracy_score(y_test, y_pred)
    print("Average accuracy = {}, f1 = {}".format(ACC, f1))
    return ACC, f1


def run_blog(model, folder=None):
    print("Load Data")
    labels = np.load(c_folder +
                     "/../../data/BlogCatalog-dataset/data/blog_labels.npy")
    eval = collections.defaultdict(list)
    if model == "gcn":
        path = GCN_FOLDER
        results = sort_nicely(glob.glob(path + "/*.npy"))
        for file in results:
            if "emb" in file and "loss" not in file:
                iter = int(file.split('/')[-1].split('_')[-1].split('.')[0])
                if iter < 1000:
                    continue
                print("*" * 50)
                print('Processing folder ', GCN_EXP)
                print('Processing file ', file.split("/")[-1])
                embedding = np.load(file)
                acc, f1 = run_one_file_blog(embedding, labels)
                eval[(folder, file)] = [acc, f1]

    elif model == "node2vec":
        path = NODE2VEC_FOLDER
        for file in sort_nicely(glob.glob(path + "/*blog*.npy")):
            print("*" * 50)
            print('Processing file ', file.split("/")[-1])
            embedding = np.load(file)
            acc, f1 = run_one_file_blog(embedding, labels)
            eval[("node2vec", file)] = [acc, f1]

    elif model == 'sdne':
        file = '/home/yba/Documents/SDNE/embeddingResult/blogcatalog3_embedding.mat'
        embedding = scipy.io.loadmat(file)['embedding']
        acc, f1 = run_one_file_blog(embedding, labels)
        eval[("sdne", file)] = [acc, f1]
    return eval


def run_blog_loss(model):
    print('Load Loss')
    loss = collections.defaultdict(float)
    if model == "gcn":
        exp = GCN_EXP
        path = GCN_FOLDER
        for file in sort_nicely(glob.glob(path + "/*.npy")):
            if "loss" in file and "emb" not in file:
                print("*" * 50)
                print('Processing folder ', GCN_EXP)
                print('Processing file ', file.split("/")[-1])
                cur_loss = np.load(file)
                print('loss = ', cur_loss)
                loss[(exp, file)] = float(cur_loss)
    return loss


def run_one_file_blog(embedding, labels):
    X_train, X_test, y_train, y_test = train_test_split(embedding, labels,
                                                        train_size=0.2)
    classif = OneVsRestClassifier(LogisticRegression(class_weight="balanced"))
    #classif = OneVsRestClassifier(SVC(kernel='linear', probability=True,
    #                                  class_weight='balanced', C=500))
    # classif = KNeighborsClassifier()
    # classif = LabelPowerset(LinearSVC(C=500))
    # classif = BinaryRelevance(classifier=SVC(), require_dense=[False, True])
    classif.fit(X_train, y_train)
    y_pred = classif.predict_proba(X_test)
    print('y_pred', y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    ACC = accuracy_score(y_test, y_pred)
    print("Average accuracy = {}, f1 = {}".format(ACC, f1))
    return ACC, f1
    #
    # f1s = []
    # accs = []
    # for i in range(len(y_train[0])):
    #     y_train_cur = y_train[:,i]
    #     y_test_cur = y_test[:,i]
    #     acc, f1 = run_model_sklearn(X_train, y_train_cur, X_test, y_test_cur)
    #     f1s.append(f1)
    #     accs.append(acc)
    #
    # print("")
    # print("Average accuracy = {}, f1 = {}".format(np.mean(accs), np.mean(f1s)))
    # return np.mean(accs), np.mean(f1s)


def run_model_sklearn(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(solver="liblinear")
    lr.fit(X_train, y_train)
    p_label = lr.predict(X_test)
    print('Result')
    ACC = accuracy_score(y_test, p_label)
    print('accuracy = ', ACC)
    f1 = f1_score(y_test, p_label, average="macro")
    print('f1_score = ', f1)

    # prob = problem(y_train, X_train)
    # param = parameter('-s 0 -c 5 -w1 20 -q -B 1')
    # m = train(prob, param)
    # p_label, p_acc, p_val = predict(y_test, X_test, m, '-b 1')
    # ACC, MSE, SCC = evaluations(y_test, p_label)
    # # print('Result')
    # print("Total number of data: {}".format(len(y_test)))
    # f1 = f1_score(y_test, p_label)
    # print('y_test nonzero count {}'.format(np.count_nonzero(y_test)))
    # print('p_label nonzero count {}'.format(np.count_nonzero(p_label)))
    # # print("accuracy = {}".format(ACC))
    # print ("f1 = {}".format(f1))
    return ACC, f1


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


def run_cora(model, folder=None):
    print('Load Data')
    eval = collections.defaultdict(list)
    adj, features, y_labels, y_val, y_truth, train_mask, val_mask, test_mask = load_data(
        'cora', 0)
    dataset = (
        adj, features, y_labels, y_val, y_truth, train_mask, val_mask,
        test_mask)
    if model == "gcn":
        if not folder:
            folders = sort_nicely(
                os.listdir(os.path.join(c_folder, "cora/gcn")))
        else:
            folders = [folder]
        for folder in folders:
            path = c_folder + "/cora/gcn/{}/intermediate".format(folder)
            for file in sort_nicely(glob.glob(path + "/*.npy")):
                if "emb" in file and "loss" not in file:
                    print("*" * 50)
                    print('Processing folder ', folder)
                    print('Processing file ', file.split("/")[-1])
                    acc, f1 = run_one_file_cora(file, dataset)
                    eval[(folder, file)] = [acc, f1]

    elif model == "node2vec":
        path = c_folder + "/cora/node2vec"
        for file in sort_nicely(glob.glob(path + "/*.npy")):
            print("*" * 50)
            print('Processing file ', file.split("/")[-1])
            acc, f1 = run_one_file_cora(file, dataset)
            eval[("node2vec", file)] = [acc, f1]
    return eval


def run_cora_loss(model, folder=None):
    print('Load Loss')
    loss = collections.defaultdict(float)
    if model == "gcn":
        if not folder:
            folders = sort_nicely(
                os.listdir(os.path.join(c_folder, "cora/gcn")))
        else:
            folders = [folder]
        for folder in folders:
            path = c_folder + "/cora/gcn/{}/intermediate".format(folder)
            for file in sort_nicely(glob.glob(path + "/*.npy")):
                if "loss" in file and "emb" not in file:
                    print("*" * 50)
                    print('Processing folder ', folder)
                    print('Processing file ', file.split("/")[-1])
                    cur_loss = np.load(file)
                    print('loss = ', cur_loss)
                    loss[(folder, file)] = float(cur_loss)
    return loss


def run_one_file_cora(file, dataset):
    embed = np.load(file)
    adj, features, y_labels, y_val, y_truth, train_mask, val_mask, test_mask = dataset
    X_train, y_train, X_test, y_test = split_data(embed, features, y_labels,
                                                  y_val, y_truth, train_mask,
                                                  val_mask, test_mask)
    y_train = process_y(y_train)
    y_test = process_y(y_test)
    acc, f1 = run_model_sklearn(X_train, y_train, X_test, y_test)
    return acc, f1


def split_data(embed, features, y_labels, y_val, y_truth, train_mask, val_mask,
               test_mask):
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
def pro_blog_label():
    labels = np.load(c_folder + "/blog/data/blog_labels.npy")
    new_labels = []
    index = 0
    dic = collections.defaultdict(int)
    for entry in labels:
        new_entry = [str(int(i)) for i in entry]
        key = "".join(new_entry)
        if key not in dic:
            dic[key] = index
            index += 1
        new_labels.append(dic[key])
    return new_labels
'''
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


if __name__ == '__main__':
    # create a data_engine object with the name of the dataset. Currently support "cora" and "blog"
    data_engine = Data_engine("blog")
    # choose the model to run. Currently support "gcn" and "node2vec"
    # if the folder name is given, it will only run the folder. If not, it will run every single folder under the path:\
    # # gcn/gcn/analysis/node_classification/your_dataset/your_data_model
    # data_engine.run("gcn")
    # #if the loss data is given, run the loss this way
    data_engine.run_with_loss('gcn')
    for (exp, file), loss in (data_engine.loss).items():
        print(file.split('/')[-1], loss)
    # # There should be one file under gcn/gcn/analysis/node_classification/your_dataset/node2vec to establish the baseline
    # data_engine.run("sdne")
    # Draw the measurements. Currently support "f1" and "acc"
    # data_engine.draw("")
