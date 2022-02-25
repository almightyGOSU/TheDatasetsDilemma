#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/06/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps

from .Compute_Similarity_Python import Compute_Similarity_Python
from .Compute_Similarity_Euclidean import Compute_Similarity_Euclidean

from enum import Enum



class SimilarityFunction(Enum):
	COSINE = "cosine"
	PEARSON = "pearson"
	JACCARD = "jaccard"
	TANIMOTO = "tanimoto"
	ADJUSTED_COSINE = "adjusted"
	EUCLIDEAN = "euclidean"



class Compute_Similarity:


	def __init__(self, dataMatrix, similarity = None, **args):
		"""
		Interface object that will call the appropriate similarity implementation
		:param dataMatrix:
		:param similarity:              the type of similarity to use, see SimilarityFunction enum
		:param args:                    other args required by the specific similarity implementation
		"""

		assert np.all(np.isfinite(dataMatrix.data)), \
			"Compute_Similarity: Data matrix contains {} non finite values".format(
				np.sum(np.logical_not(np.isfinite(dataMatrix.data))) )

		if (similarity == "euclidean"):
			# This is only available here
			self.compute_similarity_object = Compute_Similarity_Euclidean(dataMatrix, **args)

		else:

			assert not (dataMatrix.shape[0] == 1 and dataMatrix.nnz == dataMatrix.shape[1]),\
				"Compute_Similarity: data has only 1 feature (shape: {}) with dense values," \
				" vector and set based similarities are not defined on 1-dimensional dense data," \
				" use Euclidean similarity instead.".format( dataMatrix.shape )

			if (similarity is not None):
				args["similarity"] = similarity

			self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)


	def compute_similarity(self, **args):

		return self.compute_similarity_object.compute_similarity(**args)


