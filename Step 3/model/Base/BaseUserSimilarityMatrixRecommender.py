from .BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from .utilities import *



class BaseUserSimilarityMatrixRecommender(BaseSimilarityMatrixRecommender):

	def _compute_item_score(self, user_id_array, items_to_compute = None):
		"""
		URM_train and W_sparse must have the same format, CSR
		:param user_id_array:
		:param items_to_compute:
		:return:
		"""

		self._check_format()

		user_weights_array = self.W_sparse[user_id_array]

		if (items_to_compute is not None):
			item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype = np.float32) * np.inf
			item_scores_all = user_weights_array.dot(self.URM_train).toarray()
			item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
		else:
			item_scores = user_weights_array.dot(self.URM_train).toarray()

		return item_scores


