import os,glob,collections,math
from logistic_regression import *
from gcn.utils import *
import matplotlib.pyplot as plt

c_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(c_folder, "../.."))

def main():
    #tune_parameters()
    eval = run_logistic_regression()
    data = pre_draw(eval)
    draw(data)
def tune_parameters():
    # execute("rm -rfv {}/cora/node2vec/*".format(c_folder))
    #
    # print("Tuning iterations")
    # execute("mkdir {}/cora/node2vec/iter".format(c_folder))
    # for i in range(1, 51):
    #     if i == 1 or i%5 == 0:
    #         execute("python2.7 {}/../../../../node2vec/src/main.py --iter={}".format(c_folder, str(i)))
    #
    #
    # for file in glob.glob("{}/../../../../node2vec/emb/*.npy".format(c_folder)):
    #     execute("mv {} {}/cora/node2vec/iter/ ".format(file, c_folder))
    #
    # print("Tuning window-size")
    # execute("mkdir {}/cora/node2vec/win".format(c_folder))
    # for i in range(1, 51):
    #     if i == 1 or i%5 == 0:
    #         execute("python2.7 {}/../../../../node2vec/src/main.py --window-size={}".format(c_folder, str(i)))
    #
    #
    # for file in glob.glob("{}/../../../../node2vec/emb/*.npy".format(c_folder)):
    #     execute("mv {} {}/cora/node2vec/window_size/".format(file, c_folder))
    #
    # print("Tuning num-walks")
    # execute("mkdir {}/cora/node2vec/walk".format(c_folder))
    # for i in range(10, 101):
    #     if i%10 == 0:
    #         execute("python2.7 {}/../../../../node2vec/src/main.py --num-walks={}".format(c_folder, str(i)))
    #
    #
    # for file in glob.glob("{}/../../../../node2vec/emb/*.npy".format(c_folder)):
    #     execute("mv {} {}/cora/node2vec/num_walks/".format(file, c_folder))

    print("Tuning p")
    execute("mkdir {}/cora/node2vec/p".format(c_folder))
    for i in range(-3, 4):
        num = 2**i
        execute("python2.7 {}/../../../../node2vec/src/main.py --p={}".format(c_folder, str(num)))

    for file in glob.glob("{}/../../../../node2vec/emb/*.npy".format(c_folder)):
        execute("mv {} {}/cora/node2vec/p/".format(file, c_folder))

    print("Tuning q")
    execute("mkdir {}/cora/node2vec/q".format(c_folder))
    for i in range(-3, 4):
        num = 2**i
        execute("python2.7 {}/../../../../node2vec/src/main.py --q={}".format(c_folder, str(num)))

    for file in glob.glob("{}/../../../../node2vec/emb/*.npy".format(c_folder)):
        execute("mv {} {}/cora/node2vec/q/".format(file, c_folder))

def run_logistic_regression(data = "cora"):
    print('Load Data')
    eval = collections.defaultdict(list)
    adj, features, y_labels, y_val, y_truth, train_mask, val_mask, test_mask = load_data(data, 0)
    dataset = (adj, features, y_labels, y_val, y_truth, train_mask, val_mask, test_mask)
    for folder in sort_nicely(os.listdir(os.path.join(c_folder, "cora/node2vec"))):
        path = c_folder + "/cora/node2vec/{}".format(folder)
        for file in sort_nicely(glob.glob(path + "/*.npy")):
            print("*" * 50)
            print('Processing file ', file.split("/")[-1])
            acc, f1 = run_one_file_cora(file, dataset)
            eval[(folder,file)] = [acc,f1]
    return eval
def pre_draw(eval):
    data = collections.defaultdict(list)
    for folder, file in eval:
        file_name = file.split("/")[-1]
        tokens = file_name[:-4].split("_")
        num = None
        for idx, i in enumerate(tokens):
            if i == folder:
                num = float(tokens[idx+1])
                if i != "p" and i != "q":
                    num = int(num)
                else:
                    num = int(math.log(num,2))
                break
        assert(num != None)
        data[folder].append((num, eval[(folder, file)],file))
    return data
def draw(data):
    for folder in data:
        xs = []
        accs = []
        f1s = []
        setting = None
        for x, (acc, f1),file in data[folder]:
            accs.append(acc)
            f1s.append(f1)
            xs.append(x)
            if not setting:
                setting = file

        # cora_emb_iter_30_p_1_q_1_walk_10_win_1.npy
        title = gen_title(setting, folder)

        plt.figure(folder)
        plt.title("node2vec cora\n" + "default: " + title)
        plt.plot(xs, accs, label = "accs")
        if folder == "p" or folder == "q":
            folder += "log"
        plt.xlabel(folder)
        plt.ylabel("accuracy")
        plt.savefig(folder + '.png')
def gen_title(setting,folder):
    tokens = setting[:-4].split("_")[3:]
    for idx, i in enumerate(tokens):
        if i == folder:
            break
    tokens = tokens[:idx] + tokens[idx+2:]
    title = " ".join(tokens)
    return title
def execute(cmd):
    print('@@@', cmd)
    if os.system(cmd) != 0:
        exit(1)

if __name__ == "__main__":
    main()