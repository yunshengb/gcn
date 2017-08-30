import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

DIMENSION = 100

class Heatmap:

	def __init__(self, matrix):
		np.random.seed(0)
		sns.set()
		self.matrix = np.array(matrix)
		self.row = len(self.matrix)
		if self.row == 0:
			self.col = 0
		else:
			self.col = len(self.matrix[0])

	def getHeatmap(self, name, i_s = None, i_e = None, j_s = None, j_e = None):
		if not i_s:
			i_s = 0
		if not i_e:
			i_e = self.row -1
		if not j_s:
			j_s = 0
		if not j_e:
			j_e = self.col-1

		if i_s < 0 or i_e >= self.row:
			raise ValueError("No this row")
		if j_s < 0 or j_e >= self.col:
			raise ValueError("No this col")

		new_matrix = self.matrix[i_s:i_e+1, j_s:j_e+1]

		plt.figure("{}".format(name))
		ax  = sns.heatmap(new_matrix, cmap="YlGnBu")
		plt.savefig("{}.png".format(name))
		plt.show()


def predict_heatmap(filename):

	tokens = filename.split("_")
	tokens = tokens[1:]
	tokens[-1] = tokens[-1].split(".")[0]
	with open(filename) as f:
		i = 0
		for line in f:
			if i % 2 == 1:
				newline = line.strip("\n")
				new_mat = load_predict_matrix(newline)
				print new_mat.shape
				new_heatmap = Heatmap(new_mat)
				name = "predict_round_{}".format(tokens[i/2])
				new_heatmap.getHeatmap(name)
			i += 1

def truth_heatmap(filename):
	new_mat = load_truth_matrix(filename)
	print new_mat.shape

	new_heatmap = Heatmap(new_mat)
	name = "truth"
	new_heatmap.getHeatmap(name)

def load_predict_matrix(newline, dim=100):
	newline = newline.split("[")
	res_mat = []
	for line in newline:
		if line:
			numbers = line.strip("]").split(" ")
			res_mat.append(numbers)

	return np.array(res_mat,dtype =np.float32) 



def load_truth_matrix(filename, dim = 100):
	with open(filename) as f:
		res_mat = []
		i = 0
		each_row = []
		for line in f:
			numbers = line.split(" ")

			if i == dim:
				i = 0
				res_mat.append(each_row)
				each_row = []
			elif i > dim:
				raise ValueError("More numbers in a row")	

			for num in numbers:
				if num:
					each_row.append(num.strip("[").strip("]").strip("\n").strip("]"))
					i += 1
	
	res_mat.append(each_row)
	return np.array(res_mat, dtype = int)










if __name__ == "__main__":
	# uniform_data = np.random.rand(100, 100)
	# heatmap = Heatmap(uniform_data)

	# heatmap.getHeatmap(0,99,0,99, "test")

	predict_heatmap("probs_0_200_400_600_800_1000.txt")
	truth_heatmap("adj.txt")


