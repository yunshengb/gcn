import numpy as np
from utils import *
from log_reg_model import *
from sklearn.model_selection import train_test_split


def main():
	dataset = "cora"
	print('Python 3 please!')

	print('Load embeddings')
	gcn_embed = np.load('gcn_cora_emb.npy')
	print('gcn_embed',gcn_embed.shape)
	node2vec_embed = np.load('node2vec_cora_emb.npy')
	print('node2vec_embed', node2vec_embed.shape)

	print('Load ground-truths labels for cora')
	adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask= load_data(dataset,0)
	print('labels', y_train.shape)
	print('Load features and append to embeddings')
	print('features', features.shape)

	gcn_data = np.concatenate((gcn_embed, features.toarray()), axis=1)
	print ('gcn_data', gcn_data.shape)
	node2vec_data = np.concatenate((node2vec_embed, features.toarray()), axis=1)
	print ('node2vec_data', node2vec_data.shape)

	print('Perform logistic regression')
	labels = y_train

	# X_train, X_test, y_train, y_test = train_test_split(gcn_data, label, test_size=0.33, random_state=42)

	length = gcn_data.shape[0]
	X_train = gcn_data[:length*3/4,:]
	X_test = gcn_data[length*3/4:,:]
	y_train = labels[:length*3/4,:]
	y_test = labels[length*3/4:,:]
	print (y_val[-100:])
	exit(1)

	print(y_train[:20])
	print (y_test[:20])
	exit(1)

	n_epochs = 200
	learning_rate = 0.01

	lr = LogisticRegression(input=X_train, label=y_train, n_in=1633, n_out=7)
	for epoch in xrange(n_epochs):
		lr.train(lr=learning_rate)
		cost = lr.negative_log_likelihood()
		if epoch % 20 == 0:
			print ('Training epoch %d, cost is %f' % (epoch, cost))
		learning_rate *= 0.95

	print('Result')
	y_pred = lr.predict(X_test)
	print (y_pred[:20])
	print ("*" * 20)
	print (y_test[:20])



if __name__ == '__main__':
	main()