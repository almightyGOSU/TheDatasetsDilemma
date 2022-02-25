#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From: https://raw.githubusercontent.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/master/KNN/UserKNNCFRecommender.py
"""

from .Base.utilities import checkMatrix
from .Base.BaseUserSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender

from .Base.IR_feature_weighting import okapi_BM_25, TF_IDF

from .Similarity.Compute_Similarity import Compute_Similarity

import numpy as np



class UserKNNCFRecommender(BaseUserSimilarityMatrixRecommender):
	""" UserKNN recommender"""

	RECOMMENDER_NAME = "UserKNNCFRecommender"
	FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]



	def __init__(self, URM_train, verbose = True):
		super(UserKNNCFRecommender, self).__init__(URM_train, verbose = verbose)


	def fit(self, topK = 50, shrink = 100, similarity = "cosine", normalize = True, feature_weighting = "none", **similarity_args):

		self.topK = topK
		self.shrink = shrink

		if (feature_weighting not in self.FEATURE_WEIGHTING_VALUES):
			raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format( 
				self.FEATURE_WEIGHTING_VALUES, feature_weighting ))

		if (feature_weighting == "BM25"):
			self.URM_train = self.URM_train.astype(np.float32)
			self.URM_train = okapi_BM_25(self.URM_train.T).T
			self.URM_train = checkMatrix(self.URM_train, "csr")

		elif (feature_weighting == "TF-IDF"):
			self.URM_train = self.URM_train.astype(np.float32)
			self.URM_train = TF_IDF(self.URM_train.T).T
			self.URM_train = checkMatrix(self.URM_train, "csr")

		similarity = Compute_Similarity(self.URM_train.T, shrink = shrink, topK = topK, normalize = normalize, similarity = similarity, **similarity_args)

		self.W_sparse = similarity.compute_similarity()
		self.W_sparse = checkMatrix(self.W_sparse, format = "csr")


