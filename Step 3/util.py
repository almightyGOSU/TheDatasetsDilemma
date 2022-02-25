import codecs
import os
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from scipy import sparse
from tqdm import tqdm



# Cut-off for the validation metric
VALIDATION_CUTOFF = 10

# List of Cut-offs for Testing
nDictKV = {5: "nDCG@5", 10: "nDCG@10", 15: "nDCG@15", 20: "nDCG@20", 25: "nDCG@25", 50: "nDCG@50", 75: "nDCG@75", 100: "nDCG@100"}
rDictKV = {5: "Recall@5", 10: "Recall@10", 15: "Recall@15", 20: "Recall@20", 25: "Recall@25", 50: "Recall@50", 75: "Recall@75", 100: "Recall@100"}


def loadTrainData(csvFile, numItems = None, confidence = 0):
	"""
	Input:
	csvFile (string): 	the csv file contains two columns (userID, itemID).
	numItems (int): 		the number of items.

	Output:
	data (sparse matrix): each row representing a user's activities over the `numItems` items.
	"""
	df = pd.read_csv(csvFile)
	numUsers = len(df.userID.unique())
	if numItems is None: numItems = len(df.itemID.unique())

	rowIndices, colIndices = df.userID, df.itemID

	# This is for WMF, i.e. setting 'c_ui' for observed entries
	# Note: The matrix is transposed, i.e. items x users
	if (confidence != 0):

		data = sparse.csr_matrix((np.full_like(rowIndices, fill_value = confidence), (rowIndices, colIndices)),
			dtype = "float64",
			shape = (numUsers, numItems)).transpose()

	# This is for methods such as UserKNNCF, ItemKNNCF, P3alpha, RP3beta, NCF, Mult-VAE
	else:

		data = sparse.csr_matrix((np.ones_like(rowIndices), (rowIndices, colIndices)),
			dtype = "float64",
			shape = (numUsers, numItems))

	return data


def loadValidationTestData(validationFp, testFp, numItems):
	"""
	Input:
	validationFp (string): 	the csv file contains two columns (userID, itemID).
	testFp (string): 		the csv file contains two columns (userID, itemID).
	numItems (int): 		the number of items.

	Output:
	two sparse matrices: each row representing a user's activities over the `numItems` items.
	"""
	df_tr = pd.read_csv(validationFp)
	df_te = pd.read_csv(testFp)

	numValidationUsers = len(df_tr.userID.unique())
	numTestUsers = len(df_te.userID.unique())

	rows_tr, cols_tr = df_tr.userID, df_tr.itemID
	rows_te, cols_te = df_te.userID, df_te.itemID

	data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)),
		dtype = "float64",
		shape = (numValidationUsers, numItems))

	data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)),
		dtype = "float64",
		shape = (numTestUsers, numItems))

	return data_tr, data_te


def recall_at_k(X, Y, k = 1, dtype = torch.float64):
	"""
	Computing recall@k.

	Input:
	X array(n, m): the predicted matrix with each row representing a user's preference over m items.
	Y array(n, m): the ground truth matrix, assuming that all zero entries indicate no relevance.
	"""
	X = torch.tensor(X, dtype = dtype)
	Y = torch.tensor(Y, dtype = dtype)
	n = X.shape[0]

	vals, inds = torch.topk(X, k, dim = 1, sorted = False)
	X = torch.zeros_like(X, dtype = bool)
	X[torch.arange(n).reshape(-1, 1), inds] = True
	Y = Y > 0

	nnz = torch.sum(Y, dim = 1).cpu().numpy()
	return torch.sum(X * Y, dim = 1).cpu().numpy() / np.minimum(k, nnz)


def ndcg_at_k(X, Y, k = 1, dtype = torch.float64):
	"""
	Computing the NDCG@k.

	Input:
		X array(n, m): the predicted matrix with each row representing a user's preference over m items.
		Y array(n, m): the ground truth matrix, assuming that all zero entries indicate no relevance.
	"""
	X = torch.tensor(X, dtype = dtype)
	Y = torch.tensor(Y, dtype = dtype)
	n = X.shape[0]

	vals, inds = torch.topk(X, k, dim = 1)
	
	discount = 1. / torch.log2(torch.arange(2, k + 2, dtype = dtype))

	DCG = torch.sum(Y[torch.arange(n).reshape(-1, 1), inds] * discount, dim = 1)
	nnz = torch.sum(Y > 0, dim = 1).cpu().numpy()
	iDCG = torch.tensor([torch.sum(discount[0:min(i, k)]) for i in nnz])
	ndcg = (DCG / iDCG).cpu().numpy()
	return ndcg[np.logical_not(np.isnan(ndcg))]


# Get the predicted scores for all users & items
def getScores(args, mdl, trainData, numUsers, usersBatchSize, maxN = 100):

	userItemScores = []
	for startIdx in tqdm(range(0, numUsers, usersBatchSize), desc = "Predicting scores [maxN = {:,d}]".format( maxN )):

		endIdx = min(startIdx + usersBatchSize, numUsers)

		X = trainData[startIdx:endIdx]
		if sparse.isspmatrix(X):
			X = X.toarray()
		X = X.astype(np.float32)

		if (args.model == "WMF"):

			logits = np.full_like(X, fill_value = -np.inf)

			currUserBatchSize = endIdx - startIdx
			for offset in range(currUserBatchSize):

				currUserIdx = startIdx + offset

				# Get the list of (itemid, score) tuples for top 'maxN' items for this user
				item_scores = mdl.recommend(currUserIdx, trainData, N = maxN, filter_already_liked_items = True)

				for iid, item_score in item_scores:
					logits[offset, iid] = item_score

		else:

			logits = mdl._compute_item_score(np.arange(startIdx, endIdx))

		# For each user, the scores for previously consumed items are set to -inf
		logits[X.nonzero()] = -np.inf

		userItemScores.append(logits)

	return userItemScores


# Validation (for a batch of users)
def validateBatchUsers(batchUserItemScores, batchValidationData):

	# *** NOTE: It's validation nDCG @ 10!! ***
	return ndcg_at_k(batchUserItemScores, batchValidationData.toarray(), k = VALIDATION_CUTOFF)


# Testing (for a batch of users)
def testBatchUsers(batchUserItemScores, batchTestData):

	nDictResults = defaultdict(list)
	rDictResults = defaultdict(list)

	Z = batchTestData.toarray()

	for nKey in nDictKV.keys():
		nDictResults[nKey].append(ndcg_at_k(batchUserItemScores, Z, k = nKey))

	for rKey in rDictKV.keys():
		rDictResults[rKey].append(recall_at_k(batchUserItemScores, Z, k = rKey))

	return nDictResults, rDictResults


# Loads all dataset characteristics
def loadStatistics(allStatsFp):

	if (os.path.exists(allStatsFp)):

		with codecs.open(allStatsFp, "r", encoding = "utf-8", errors = "ignore") as inFile:
			contents = inFile.readlines()

		datasetStats = defaultdict(list)
		for line in contents:

			elements = line.strip().split(", ")

			# dataset
			dataset = elements[0].strip()

			# (0) numUsers, 		(1) numItems, 		(2) numInteractions, 	(3) density
			# (4) userMin, 			(5) userMax, 		(6) userAvg, 			(7) itemMin, 		(8) itemMax, 		(9) itemAvg
			# (10) spaceSizeLog, 	(11) shapeLog, 		(12) densityLog, 		(13) userGini, 		(14) itemGini

			stats = [x.strip() for x in elements[1:]]
			stats = [int(float(x)) if int(float(x)) == float(x) else float(x) for x in stats]

			datasetStats[dataset] = stats

	return datasetStats


