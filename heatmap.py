import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

	def getHeatmap(self, i_s, i_e, j_s, j_e, name):
		if i_s < 0 or i_e >= self.row:
			raise ValueError("No this row")
		if j_s < 0 or j_e >= self.col:
			raise ValueError("No this col")

		new_matrix = self.matrix[i_s:i_e+1, j_s:j_e+1]

		plt.figure()
		ax  = sns.heatmap(new_matrix, vmin = 0.0, vmax = 1.0, cmap="YlGnBu")
		plt.savefig("{}.png".format(name))
		plt.show()


if __name__ == "__main__":
	uniform_data = np.random.rand(10, 12)
	heatmap = Heatmap(uniform_data)

	heatmap.getHeatmap(1,5,0,5, "test")




