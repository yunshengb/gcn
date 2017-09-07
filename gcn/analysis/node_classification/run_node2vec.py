import os,glob
from logistic_regression import *

current_folder = os.path.dirname(os.path.realpath(__file__))

def main():
    #gen_iter()
    run_dataset("cora")

def gen_iter():

    for i in range(1, 51):
        if i == 1 or i%10 == 0:
            execute("python2.7 {}/../../../../node2vec/src/main.py --iter={}".format(current_folder, str(i)))
    execute("rm -rfv npy_files/*")
    for file in glob.glob("{}/../../../../node2vec/emb/*.npy".format(current_folder)):
        execute("mv {} {}/npy_files/ ".format(file, current_folder))

def execute(cmd):
    print('@@@', cmd)
    if os.system(cmd) != 0:
        exit(1)


if __name__ == "__main__":
    main()