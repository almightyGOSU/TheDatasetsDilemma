import numpy as np
import scipy.sparse as sps
import time
import os



def checkMatrix(X, format = "csc", dtype = np.float32):
	"""
	This function takes a matrix as input and transforms it into the specified format.
	The matrix in input can be either sparse or ndarray.
	If the matrix in input has already the desired format, it is returned as-is
	the dtype parameter is always applied and the default is np.float32
	:param X:
	:param format:
	:param dtype:
	:return:
	"""

	if isinstance(X, np.ndarray):
		X = sps.csr_matrix(X, dtype = dtype)
		X.eliminate_zeros()
		return checkMatrix(X, format = format, dtype = dtype)

	elif format == 'csc' and not isinstance(X, sps.csc_matrix):
		return X.tocsc().astype(dtype)
	elif format == 'csr' and not isinstance(X, sps.csr_matrix):
		return X.tocsr().astype(dtype)
	elif format == 'coo' and not isinstance(X, sps.coo_matrix):
		return X.tocoo().astype(dtype)
	elif format == 'dok' and not isinstance(X, sps.dok_matrix):
		return X.todok().astype(dtype)
	elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
		return X.tobsr().astype(dtype)
	elif format == 'dia' and not isinstance(X, sps.dia_matrix):
		return X.todia().astype(dtype)
	elif format == 'lil' and not isinstance(X, sps.lil_matrix):
		return X.tolil().astype(dtype)

	elif format == 'npy':
		if sps.issparse(X):
			return X.toarray().astype(dtype)
		else:
			return np.array(X)

	else:
		return X.astype(dtype)


def similarityMatrixTopK(item_weights, k = 100, verbose = False):
	"""
	The function selects the TopK most similar elements, column-wise

	:param item_weights:
	:param forceSparseOutput:
	:param k:
	:param verbose:
	:param inplace: Default True, WARNING matrix will be modified
	:return:
	"""

	assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

	start_time = time.time()

	if verbose:
		print("Generating topK matrix")

	nitems = item_weights.shape[1]
	k = min(k, nitems)

	# for each column, keep only the top-k scored items
	sparse_weights = not isinstance(item_weights, np.ndarray)

	# iterate over each column and keep only the top-k similar items
	data, rows_indices, cols_indptr = [], [], []

	if sparse_weights:
		item_weights = checkMatrix(item_weights, format = 'csc', dtype = np.float32)
	else:
		column_row_index = np.arange(nitems, dtype = np.int32)

	for item_idx in range(nitems):

		cols_indptr.append(len(data))

		if sparse_weights:
			start_position = item_weights.indptr[item_idx]
			end_position = item_weights.indptr[item_idx+1]

			column_data = item_weights.data[start_position:end_position]
			column_row_index = item_weights.indices[start_position:end_position]

		else:
			column_data = item_weights[:,item_idx]

		non_zero_data = column_data!=0

		idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
		top_k_idx = idx_sorted[-k:]

		data.extend(column_data[non_zero_data][top_k_idx])
		rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

	cols_indptr.append(len(data))

	# During testing CSR is faster
	W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape = (nitems, nitems), dtype = np.float32)

	if verbose:
		print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

	return W_sparse


def seconds_to_biggest_unit(time_in_seconds, data_array = None):

	conversion_factor = [
		("sec", 60),
		("min", 60),
		("hour", 24),
		("day", 365),
	]

	terminate = False
	unit_index = 0

	new_time_value = time_in_seconds
	new_time_unit = "sec"

	while not terminate:

		next_time = new_time_value / conversion_factor[unit_index][1]

		if next_time >= 1.0:
			new_time_value = next_time

			if data_array is not None:
				data_array /= conversion_factor[unit_index][1]

			unit_index += 1
			new_time_unit = conversion_factor[unit_index][0]

		else:
			terminate = True

	if data_array is not None:
		return new_time_value, new_time_unit, data_array

	else:
		return new_time_value, new_time_unit


