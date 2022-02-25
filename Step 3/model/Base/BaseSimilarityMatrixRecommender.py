from .BaseRecommender import BaseRecommender
from .utilities import *



class BaseSimilarityMatrixRecommender(BaseRecommender):
	"""
	This class refers to a BaseRecommender KNN which uses a similarity matrix, it provides two function to compute item's score
	bot for user-based and Item-based models as well as a function to save the W_matrix
	"""

	def __init__(self, URM_train, verbose=True):

		super(BaseSimilarityMatrixRecommender, self).__init__(URM_train, verbose = verbose)

		self._URM_train_format_checked = False
		self._W_sparse_format_checked = False


	def _check_format(self):

		if (not self._URM_train_format_checked):

			if self.URM_train.getformat() != "csr":
				self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("URM_train", "csr"))

			self._URM_train_format_checked = True

		if (not self._W_sparse_format_checked):

			if self.W_sparse.getformat() != "csr":
				self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("W_sparse", "csr"))

			self._W_sparse_format_checked = True


