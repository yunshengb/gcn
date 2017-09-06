import numpy as np
from utils import *
from log_reg_model import *
from sklearn.model_selection import train_test_split
import tensorflow as tf


def run_dataset(dataset):
    print('Python 3 please!')
    print('Load embeddings')
    embeddings = ['gcn_cora_emb.npy','node2vec_cora_emb.npy']
    for embedding in embeddings:
        analyze_embedding(embedding)

def analyze_embedding(embedding):
    embed = np.load(embedding)
    print ("*" * 10)
    print('Processing file ', embedding)
    print('Load ground-truths labels for cora')
    adj, features, y_labels, y_val, y_truth, train_mask, val_mask, test_mask = load_data(dataset, 0)
    print('labels', y_labels.shape)
    print('Load features and append to embeddings')
    print('features', features.shape)
    data = np.concatenate((embed, features.toarray()), axis=1)
    print('data', data.shape)
    
    print('Perform logistic regression')
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

    n_epochs = 200
    learning_rate = 0.01

    lr = LogisticRegression(input=X_train, label=y_train, n_in=X_train.shape[1], n_out=y_train.shape[1])
    for epoch in range(n_epochs):
        lr.train(lr=learning_rate, L2_reg = 1.0)
        cost = lr.negative_log_likelihood()
        if epoch % 20 == 0:
            print('Training epoch %d, cost is %f' % (epoch, cost))
        learning_rate *= 0.95

    print('Result')
    y_pred = lr.predict(X_test)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    sess = tf.Session()
    acc_result = sess.run(accuracy_all)
    acc = sum(acc_result)/len(acc_result)
    print(acc)
    

if __name__ == '__main__':
    dataset = "cora"
    run_dataset(dataset)
