from .BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from .utilities import *



class BaseItemSimilarityMatrixRecommender(BaseSimilarityMatrixRecommender):

	def _compute_item_score(self, user_id_array, items_to_compute = None):
		"""
		URM_train and W_sparse must have the same format, CSR
		:param user_id_array:
		:param items_to_compute:
		:return:
		"""

		self._check_format()

		user_profile_array = self.URM_train[user_id_array]

		if (items_to_compute is not None):
			item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype = np.float32) * np.inf
			item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
			item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
		else:
			item_scores = user_profile_array.dot(self.W_sparse).toarray()

		return item_scores


