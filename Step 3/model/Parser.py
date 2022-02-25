import argparse



def parse_args():

	parser = argparse.ArgumentParser()

	parser.add_argument("-d", 			dest = "dataset", 		type = str, 	default = "ML-100K",
		help = "Dataset, e.g. ML-100K, ... (Default: ML-100K)")
	parser.add_argument("-m", 			dest = "model", 		type = str, 	default = "UserKNNCF",
		help = "Model Name, e.g. UserKNNCF|ItemKNNCF|RP3beta|WMF (Default: UserKNNCF)")


	# For UserKNNCF, ItemKNNCF
	parser.add_argument("-KNNCF_sim", 			dest = "KNNCF_similarity", 		type = str, 	default = "cosine",
		help = "Similarity Function, e.g. cosine|pearson|jaccard|euclidean|... (Default: cosine)")

	parser.add_argument("-KNNCF_topK", 			dest = "KNNCF_topK", 			type = int, 	default = 5,
		help = "Neighborhood Size 'K' (Default: 5)")

	parser.add_argument("-KNNCF_shrink", 		dest = "KNNCF_shrink", 			type = int, 	default = 0,
		help = "Shrink Term 'h', used to lower the similarity between items (or users) having only few interactions (Default: 0)")

	parser.add_argument("-KNNCF_norm", 			dest = "KNNCF_normalize", 		type = int, 	default = 0,
		help = "Similarity may or not be normalized via the product of vector norms (Default: 0, i.e. Disabled)")

	parser.add_argument("-KNNCF_feat_weight", 	dest = "KNNCF_feat_weight", 	type = str, 	default = "none",
		help = "Feature Weighting, e.g. none|BM25|TF-IDF (Default: none)")


	# For RP3beta
	parser.add_argument("-graph_topK", 			dest = "graph_topK", 			type = int, 	default = 100,
		help = "Top 'K' (Default: 100)")

	parser.add_argument("-graph_alpha", 		dest = "graph_alpha", 			type = float, 	default = 1.0,
		help = "Alpha (Default: 1.0)")

	parser.add_argument("-graph_beta", 			dest = "graph_beta", 			type = float, 	default = 0.6,
		help = "Beta (Default: 0.6)")

	parser.add_argument("-graph_norm", 			dest = "graph_norm", 			type = int, 	default = 0,
		help = "Normalize Similarity (Default: 0, i.e. False)")


	# For WMF
	parser.add_argument("-WMF_c_ui", 			dest = "WMF_c_ui", 				type = int, 	default = 2,
		help = "c_ui; The confidence value, e.g. {2, 5, 10, 30, 50, 100} (Default: 2)")

	parser.add_argument("-WMF_factors", 		dest = "WMF_factors", 			type = int, 	default = 100,
		help = "Number of User & Item Factors, e.g. {100, 200} (Default: 100)")

	parser.add_argument("-WMF_reg", 			dest = "WMF_reg", 				type = float, 	default = 0.01,
		help = "Regularization Weight (Default: 0.01)")

	parser.add_argument("-WMF_iterations", 		dest = "WMF_iterations", 		type = int, 	default = 15,
		help = "Number of Iterations (Default: 15)")


	# Miscellaneous
	parser.add_argument("-rs", 		dest = "random_seed", 			type = int, default = 1337, help = "Random Seed (Default: 1337)")
	parser.add_argument("-vb", 		dest = "verbose", 				type = int, default = 0, 	help = "Show debugging/miscellaneous information? (Default: 0, i.e. Disabled)")

	return parser.parse_args()


