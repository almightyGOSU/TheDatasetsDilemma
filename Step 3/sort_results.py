import time
from tqdm import tqdm
import os
import codecs
import argparse
import statistics as stats

from collections import defaultdict, OrderedDict
from scipy.stats import rankdata

from utilities_results import *

import numpy as np



# Sort and save the extracted results
def sortAndSave(args, BASE_FOLDER, lstResults, hyperParamNamesHeader):

	# List of cut-offs
	lst_Ks = [5, 10, 15, 20, 25, 50, 75, 100]

	# Output
	fp_sorted_results = 	"{}___sorted_results___.txt".format( BASE_FOLDER )
	fp_best_result = 		"{}___best_result___.txt".format( BASE_FOLDER )

	#		0					1			2			3			4		 5
	# processedCommand, validationScore, bestEpoch, lst_ndcg, lst_recall, filename

	# A newer file will take precedene in the event of a tie 
	lstResults = sorted(lstResults, key = lambda result: result[5], reverse = True)

	# Sort by the 'validation' nDCG@10 (in DESCENDING order)
	lstResults = sorted(lstResults, key = lambda result: result[1], reverse = True)

	with codecs.open(fp_sorted_results, 'w', encoding = 'utf-8', errors = 'ignore') as outFile:

		outFile.write("Reporting the nDCG & Recall for '{}' on '{}'..\n".format( args.model, args.dataset ))
		outFile.write("[# of results: {}]\n\n".format( len(lstResults) ))

		outFile.write("{:^8s} || {:^33s} || {:^33s} || {}     {}     {}\n".format(
			"Val nDCG", "Test nDCG", "Test Recall",
			"(╯°□°)╯︵ ┻━┻", "(╯°□°)╯︵ ┻━┻", "(╯°□°)╯︵ ┻━┻" ))

		outFile.write("{:<8s} || {:^33s} || {:^33s} || {:19s} || {}\n".format(
			"@ 10",
			" | ".join(["{:<6s}".format( "@ {}".format( K ) ) for K in lst_Ks[:4]]),
			" | ".join(["{:<6s}".format( "@ {}".format( K ) ) for K in lst_Ks[:4]]),
			"Filename", hyperParamNamesHeader ))

		for hyperParamValues, validationScore, bestEpoch, lst_ndcg, lst_recall, filename in lstResults:

			outFile.write("{:<8s} || {:33s} || {:33s} || {:19s} || {}\n".format(
				"{:7.5f}".format( validationScore ),
				" | ".join(["{:6.4f}".format( ndcg[0] ) for ndcg in lst_ndcg[:4]]),
				" | ".join(["{:6.4f}".format( rec[0] ) for rec in lst_recall[:4]]),
				filename[:19],
				hyperParamValues ))

	print("\n{:20s} (D: {}, M: {}): \"{}\"".format(
		"Sorted Results", args.dataset, args.model.replace("KNNCF", "KNN").replace("MultiVAE", "Mult-VAE"), fp_sorted_results ))

	with codecs.open(fp_best_result, 'w', encoding = 'utf-8', errors = 'ignore') as outFile:

		outFile.write("{:<16s} {}\n".format( "Filename:", lstResults[0][5] ))
		outFile.write("{:<16s} {}\n\n".format( "Command:", lstResults[0][0] ))

		outFile.write("{:<16s} {:.5f}\n\n".format( "Val nDCG@10:", lstResults[0][1] ))

		for (K, ndcg_at_K) in zip(lst_Ks, lstResults[0][3]):
			outFile.write("{:<16s} {:.5f} ({:.5f})\n".format( "Test nDCG@{}:".format( K ), ndcg_at_K[0], ndcg_at_K[1] ))

		outFile.write("\n")

		for (K, rec_at_K) in zip(lst_Ks, lstResults[0][4]):
			outFile.write("{:<16s} {:.5f} ({:.5f})\n".format( "Test Recall@{}:".format( K ), rec_at_K[0], rec_at_K[1] ))

		outFile.write("\n")

	print("{:20s} (D: {}, M: {}): \"{}\"".format(
		"Best Result", args.dataset, args.model.replace("KNNCF", "KNN").replace("MultiVAE", "Mult-VAE"), fp_best_result ))


# Process results for the specified dataset & the specified model
def processResults(args, BASE_FOLDER, result_files):

	lstResults = []
	bBuggy = False
	hyperParamNamesHeader = None

	for result_file in sorted(result_files):

		with codecs.open(result_file, 'r', encoding = 'utf-8', errors = 'ignore') as inFile:

			# Filename
			filename = result_file.split("/")
			filename = filename[-1].replace("-logs.txt", "")

			hyperParamNames, hyperParamValues, validationScore, bestEpoch, lst_ndcg, lst_recall = processDoc(inFile, args)

		# Possibly incomplete file
		if (bestEpoch is None):

			# Just to create a gap before the first 'buggy file', i.e. incomplete
			if (not bBuggy):
				bBuggy = True
				print("")

			print("\t***** Buggy file: {} (D: {}, M: {}) *****".format(
				filename, args.dataset, args.model.replace("KNNCF", "KNN").replace("MultiVAE", "Mult-VAE") ))
			continue

		hyperParamNamesHeader = hyperParamNames
		lstResults.append( [hyperParamValues, validationScore, bestEpoch, lst_ndcg, lst_recall, filename] )

	# Short Pause (so that we don't 'miss' the errors)
	if (bBuggy):
		time.sleep(3.00)

	# In case there are no valid results
	if (len(lstResults) == 0):
		print("\n[ERROR] \"{}\" does not contain any valid results!!\n".format( BASE_FOLDER ))
		exit()

	# Sort and save all results
	sortAndSave(args, BASE_FOLDER, lstResults, hyperParamNamesHeader)



if (__name__ == '__main__'):

	parser = argparse.ArgumentParser()

	parser.add_argument("-d", 		dest = "dataset", 			type = str, 	default = "ML-100K",
		help = "Dataset for Running Experiments (Default: ML-100K)")

	parser.add_argument("-m", 		dest = "model", 			type = str, 	default = "UserKNNCF",
		help = "Model Name, e.g. UserKNNCF|ItemKNNCF|RP3beta|WMF|MultiVAE (Default: UserKNNCF)")

	args = parser.parse_args()

	# Clean
	args.dataset = args.dataset.strip()
	args.model = args.model.strip()


	# Input
	BASE_FOLDER = "./logs/{}/{}/".format( args.dataset, args.model )
	if (os.path.isdir(BASE_FOLDER) == False):
		print("\n[ERROR] \"{}\" is not a valid directory!!\n".format( BASE_FOLDER ))
		exit()

	result_files = [os.path.join(BASE_FOLDER, f) for f in os.listdir(BASE_FOLDER) if f.endswith("-logs.txt")]


	# Process results for the specified dataset & the specified model
	processResults(args, BASE_FOLDER, result_files)
	print("")


