from .utilities import *



class BaseRecommender(object):
	"""Abstract BaseRecommender"""

	RECOMMENDER_NAME = "Recommender_Base_Class"

	def __init__(self, URM_train, verbose = True):

		super(BaseRecommender, self).__init__()

		self.URM_train = checkMatrix(URM_train.copy(), "csr", dtype = np.float32)
		self.URM_train.eliminate_zeros()

		self.n_users, self.n_items = self.URM_train.shape
		self.verbose = verbose

		self.filterTopPop = False
		self.filterTopPop_ItemsID = np.array([], dtype=np.int)

		self.items_to_ignore_flag = False
		self.items_to_ignore_ID = np.array([], dtype=np.int)

		self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

		if (self._cold_user_mask.any()):
			self._print("URM Detected {:,d} ({:.2f} %) cold users.".format(
				self._cold_user_mask.sum(), self._cold_user_mask.sum() / self.n_users * 100))


		self._cold_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) == 0

		if (self._cold_item_mask.any()):
			self._print("URM Detected {:,d} ({:.2f} %) cold items.".format(
				self._cold_item_mask.sum(), self._cold_item_mask.sum() / self.n_items * 100))


	def _get_cold_user_mask(self):
		return self._cold_user_mask

	def _get_cold_item_mask(self):
		return self._cold_item_mask


	def _print(self, string):
		if self.verbose:
			print("{}: {}".format(self.RECOMMENDER_NAME, string))

	def fit(self):
		pass

	def get_URM_train(self):
		return self.URM_train.copy()

	def set_URM_train(self, URM_train_new, **kwargs):

		assert self.URM_train.shape == URM_train_new.shape, "{}: set_URM_train old and new URM train have different shapes".format(self.RECOMMENDER_NAME)

		if len(kwargs) > 0:
			self._print("set_URM_train keyword arguments not supported for this recommender class. Received: {}".format(kwargs))

		self.URM_train = checkMatrix(URM_train_new.copy(), 'csr', dtype = np.float32)
		self.URM_train.eliminate_zeros()

		self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

		if self._cold_user_mask.any():
			self._print("Detected {} ({:.2f} %) cold users.".format(
				self._cold_user_mask.sum(), self._cold_user_mask.sum()/len(self._cold_user_mask)*100))



	def set_items_to_ignore(self, items_to_ignore):
		self.items_to_ignore_flag = True
		self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

	def reset_items_to_ignore(self):
		self.items_to_ignore_flag = False
		self.items_to_ignore_ID = np.array([], dtype=np.int)


	#########################################################################################################
	##########                                                                                     ##########
	##########                     COMPUTE AND FILTER RECOMMENDATION LIST                          ##########
	##########                                                                                     ##########
	#########################################################################################################


	def _remove_TopPop_on_scores(self, scores_batch):
		scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
		return scores_batch


	def _remove_custom_items_on_scores(self, scores_batch):
		scores_batch[:, self.items_to_ignore_ID] = -np.inf
		return scores_batch


	def _remove_seen_on_scores(self, user_id, scores):

		assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

		seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

		scores[seen] = -np.inf
		return scores


	def _compute_item_score(self, user_id_array, items_to_compute = None):
		"""

		:param user_id_array:       array containing the user indices whose recommendations need to be computed
		:param items_to_compute:    array containing the items whose scores are to be computed.
										If None, all items are computed, otherwise discarded items will have as score -np.inf
		:return:                    array (len(user_id_array), n_items) with the score.
		"""
		raise NotImplementedError("BaseRecommender: compute_item_score not assigned for current recommender, unable to compute prediction scores")


