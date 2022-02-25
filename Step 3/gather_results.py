from utilities_results import *

import codecs
import argparse
import time
import numpy as np
import subprocess

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import to_rgba

from scipy.stats import rankdata, ttest_ind_from_stats

from util import loadStatistics
import pandas as pd



if (__name__ == '__main__'):

	parser = argparse.ArgumentParser()

	parser.add_argument("-r", 			dest = "refresh", 			type = int, 	default = 1, 	help = "1: Refresh the results! (Default: 1)")
	parser.add_argument("-p", 			dest = "plot", 				type = int, 	default = 1, 	help = "1: Plot the gathered results! (Default: 1)")

	parser.add_argument("-nc", 			dest = "num_clusters", 		type = int, 	default = 5, 	help = "Number of clusters (Default: 5)")

	args = parser.parse_args()


	# Input
	BASE_FOLDER = "./logs/"
	DATASETS_FOLDER = "../Datasets/Preprocessed/"


	# Load dataset characteristics
	ALL_STATS_FILE = "../Datasets/characteristics_all.txt"
	datasetStats = loadStatistics(ALL_STATS_FILE)


	# Input (Clustering Information)
	clusteringFp = "../Step 2/Sampled Datasets ({} Clusters).txt".format( args.num_clusters )

	# Read clustering file
	with codecs.open(clusteringFp, "r", encoding = "utf-8", errors = "ignore") as inFile:
		contents = inFile.readlines()

	# List of lists (Each sublist contains the 'representative' datasets within that cluster)
	lstClusterDatasets = []

	tmp = None
	for line in contents:

		line = line.strip()

		if (not line):
			pass

		elif (line.startswith("Cluster ")):

			if tmp:
				lstClusterDatasets.append(tmp)
			tmp = []

		else:

			dataset = line.split("  ", maxsplit = 1)[1].strip()
			tmp.append(dataset)

	# For the last cluster
	if tmp:
		lstClusterDatasets.append(tmp)

	# A list of all datasets
	lstAllDatasets = [item for sublist in lstClusterDatasets for item in sublist]

	print("\n# of Clusters: {:,d}".format( len(lstClusterDatasets) ))
	for clusterIdx, clusterDatasets in enumerate(lstClusterDatasets):
		print("[Cluster {:d}] {}".format( clusterIdx + 1, ", ".join(clusterDatasets) ))


	# Output
	resultsSummaryFp = "{}___results_summary___.txt".format( BASE_FOLDER )


	# List of cut-offs
	lst_Ks = [5, 10, 15, 20, 25, 50, 75, 100]

	# For overall comparison (across models, for each dataset)
	dataset_results_ndcg = defaultdict(dict)
	dataset_results_recall = defaultdict(dict)
	dataset_results_ndcg_stdDev = defaultdict(dict)
	dataset_results_recall_stdDev = defaultdict(dict)

	MODELS = ["UserKNN", "ItemKNN", "RP3beta", "WMF", "Mult-VAE"]


	# Number of users for each dataset
	DATASET_2_DATA_POINTS = defaultdict(int)
	for dataset in lstAllDatasets:

		currDatasetFolder = "{}{}/".format( DATASETS_FOLDER, dataset )

		allUniqueUserIDs = np.genfromtxt(os.path.join(currDatasetFolder, "users.txt"), dtype = "str")
		numUsers = len(allUniqueUserIDs)

		DATASET_2_DATA_POINTS[dataset] = numUsers


	# Process all available experimental results
	lstDatasetFolders = [os.path.join(BASE_FOLDER, f) for f in os.listdir(BASE_FOLDER) if (
		not f.startswith("___results_summary___") and not f.startswith("___LaTeX___Table___"))]

	for datasetFolder in sorted(lstDatasetFolders):

		# Current Dataset
		currDataset = datasetFolder.split("/")[-1].strip()

		if (currDataset not in lstAllDatasets):
			continue

		lstModelFolders = [os.path.join(datasetFolder, f) for f in os.listdir(datasetFolder)]

		print("")
		for modelFolder in sorted(lstModelFolders):

			# Current Model
			currModel = modelFolder.split("/")[-1].strip()

			print("[D: {}, M: {}] >>>>> Processing Results...".format(
				currDataset, currModel.replace("KNNCF", "KNN").replace("MultiVAE", "Mult-VAE") ))

			# [Optional] Refresh results
			if (args.refresh):
				subprocess.call("exec python3 sort_results.py -d \"{}\" -m \"{}\"".format( currDataset, currModel ), shell = True)

			currPath = "{}{}/{}/".format( BASE_FOLDER, currDataset, currModel )
			fp_best_result, fp_best_result_logs = "{}{}".format( currPath, "___best_result___.txt" ), None

			if (not os.path.isfile(fp_best_result)):
				continue

			# Retrieve filename for the best result
			with codecs.open(fp_best_result, 'r', encoding = 'utf-8', errors = 'ignore') as inFile:
				fp_best_result_logs = "{}{}-logs.txt".format( currPath, inFile.readline().strip().split(":")[1].strip())

			# Read & process the best result
			if (fp_best_result_logs):
				with codecs.open(fp_best_result_logs, 'r', encoding = 'utf-8', errors = 'ignore') as inFile:
					args.dataset, args.model, = currDataset, currModel
					_, _, _, _, lst_ndcg, lst_recall = processDoc(inFile, args)


			# Shorten name
			currModel = currModel.replace("KNNCF", "KNN").replace("MultiVAE", "Mult-VAE")

			# For the overall comparison of models on each dataset
			for (K, ndcg_at_K) in zip(lst_Ks, lst_ndcg):
				dataset_results_ndcg[currDataset][(currModel, K)] = ndcg_at_K[0]
				dataset_results_ndcg_stdDev[currDataset][(currModel, K)] = ndcg_at_K[1]

			for (K, recall_at_K) in zip(lst_Ks, lst_recall):
				dataset_results_recall[currDataset][(currModel, K)] = recall_at_K[0]
				dataset_results_recall_stdDev[currDataset][(currModel, K)] = recall_at_K[1]


	# For gathering results (to compare different models)...
	# Recall @ 10, nDCG @ 10
	K = 10

	headerStr = "{:<30s} || {:^140s} ||\n"
	sepStr = "{:<30s} || {:^140s} ||\n".format( "=" * 30, "=" * 140 )

	for metric, results in [("nDCG", dataset_results_ndcg), ("Rec", dataset_results_recall)]:

		currFp = resultsSummaryFp.replace(".txt", "{:s}_{:d}.txt".format( metric, K ))
		with codecs.open(currFp, 'w', encoding = 'utf-8', errors = 'ignore') as outFile:

			for clusterIdx, clusterDatasets in enumerate(lstClusterDatasets):

				outFile.write(headerStr.format( "Cluster {:d}".format( clusterIdx + 1 ), "Test {:s} @ {:d}".format( metric, K ) ))
				outFile.write(sepStr)
				outFile.write(headerStr.format( "Datasets", " | ".join(["{:<12s} {:<7s}".format(
					"#{:d} Model".format( idx + 1 ), "Score" ) for idx, _ in enumerate(MODELS)]) ))

				for clusterDataset in clusterDatasets:

					if (clusterDataset not in results.keys()):
						continue

					datasetResults = results[clusterDataset]

					lstModelScores = []
					for model in MODELS:
						lstModelScores.append([model, datasetResults[(model, K)] if (model, K) in datasetResults.keys() else 0.0])

					lstModelScores.sort(key = lambda x: x[1], reverse = True)

					outFile.write(headerStr.format( clusterDataset, " | ".join(["{:<12s} {:<7s}".format(
						model, "{:.5f}".format( score ) ) for (model, score) in lstModelScores]) ))

				outFile.write(sepStr + "\n")

	print("\n++++ ++++ ++++ Overall Comparison (of various models, for each dataset) saved to \"{}\"\n\n".format( resultsSummaryFp ))


	# [Optional] Plotting the results
	if (args.plot):

		ANNO_FONT_SIZE = 15
		LEGEND_FONT_SIZE = 17.5

		# Recall @ 10, nDCG @ 10
		K = 10
		for metric, results in [("nDCG", dataset_results_ndcg), ("Rec", dataset_results_recall)]:

			# Bar Plot
			currFp = resultsSummaryFp.replace(".txt", "{:s}_{:d}__bar.png".format( metric, K ))

			print("Plotting Bar Plot for Test {:s}@{:d}...".format( metric, K ))

			colors = ["red", "limegreen", "deepskyblue", "yellow", "blueviolet", "peru", "orchid", "cyan", "greenyellow", "darkorange"]
			hatches = ["///", "\\\\\\", "...", "**", "XXX", "OO", "---", "oooo"]

			# Default fig size is [6.4, 4.8] (in inches, not pixels)
			mainfig = plt.figure(figsize = (24.0, 10.0), dpi = 300)

			def plot_metric(plt, results, currMetric = "nDCG"):

				for clusterIdx, clusterDatasets in enumerate(lstClusterDatasets):

					# 1 row, 'args.num_clusters' columns
					ax = plt.subplot(1, args.num_clusters, (clusterIdx + 1))

					lstEntries, lstRanks = [], []
					for clusterDataset in clusterDatasets:

						if (clusterDataset not in results.keys()):
							continue

						datasetResults = results[clusterDataset]

						datasetName = clusterDataset.strip()
						if (datasetName.startswith("Amazon (")):
							datasetName = "Amazon\n{:s}".format( datasetName.replace("Amazon ", "") )

						lstCurrEntries = []
						for model in MODELS:

							lstCurrEntries.append([
								datasetName,
								model,
								datasetResults[(model, K)] if (model, K) in datasetResults.keys() else 0.0
							])

						# Note: Negation, i.e. * -1
						ranks = rankdata([(x[2] * -1) for x in lstCurrEntries], method = "min").astype(int)

						lstCurrRanks = []
						for model, rank in zip(MODELS, ranks):

							lstCurrRanks.append([
								datasetName,
								model,
								rank
							])

						lstEntries.extend(lstCurrEntries)
						lstRanks.extend(lstCurrRanks)

					if (not lstEntries or not lstRanks):
						continue

					df = pd.DataFrame(lstEntries, columns = ["Dataset", "Model", "Score"])
					df = df.pivot("Dataset", "Model", "Score")
					df = df.reindex(columns = MODELS)
					df.plot(kind = "bar", ax = ax, width = 0.95)

					# Set hatch pattern
					# From: https://stackoverflow.com/questions/22833404/how-do-i-plot-hatched-bars-using-pandas/48507993
					bars = ax.patches

					barHatches = []
					barColors = []

					for idx, _ in enumerate(MODELS):
						barHatches += [hatches[idx]] * len(df)
						barColors += [colors[idx]] * len(df)

					for bar, hatch, color in zip(bars, barHatches, barColors):
						bar.set_hatch(hatch)
						bar.set_fc(color)
						bar.set_ec("black")
						bar.set_lw(0.5)

					# Note the difference from above
					# Reason: ax.patches is returned in a weird order...
					ranksDf = pd.DataFrame(lstRanks, columns = ["Dataset", "Model", "Rank"])
					ranksDf = ranksDf.pivot("Model", "Dataset", "Rank")
					ranksDf = ranksDf.reindex(index = MODELS)

					for p, rank in zip(ax.patches, ranksDf.to_numpy().flatten()):
						ax.annotate("x" if (p.get_height() == 0) else str(rank),
							((p.get_x() + p.get_width() / 2), p.get_height()),
							ha = "center", va = "bottom",
							fontsize = ANNO_FONT_SIZE,
							fontweight = "normal" if (p.get_height() == 0) else "bold",
							color = "#FF0000" if (p.get_height() == 0) else "#000000")

					# Find the minimum & maxmimum score
					minScore, maxScore = np.min([x[2] for x in lstEntries]), np.max([x[2] for x in lstEntries])

					# Calculate y Limits (Standardized across all clusters)
					y_limit_btm = 0.0 # max(0.0, minScore - 0.025)
					y_limit_top = min(1.0, maxScore + ((maxScore - minScore) * 0.25))

					if (clusterIdx in [2, 3]):
						y_limit_top += 0.0025
					elif (clusterIdx in [0, 1]):
						y_limit_top -= 0.015
					elif (clusterIdx in [4] and metric == "Rec"):
						y_limit_top -= 0.015

					# Adjust y Limits
					ax.set_ylim([y_limit_btm, y_limit_top])

					# Set x-Label
					ax.set_xlabel("Cluster {:d}".format( clusterIdx + 1 ),
						fontsize = 24, fontweight = "bold",
						labelpad = 16)
					ax.xaxis.set_label_position('top')

					ax.tick_params(axis = "x", length = 8, labelsize = 18, pad = 2)
					plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")

					ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
					ax.tick_params(axis = "y", length = 8, labelsize = 18, pad = 2)
					ax.tick_params(axis = "y", which = "minor", length = 5)
					plt.setp(ax.get_yticklabels(), fontstyle = "italic")

					# if (clusterIdx > 0):
					# 	ax.set_yticks([])

					y_limit_btm, y_limit_top = ax.get_ylim()

					for yminor in ax.yaxis.get_minorticklocs():
						ax.axhline(y = yminor, ls = '--', alpha = 0.5, linewidth = 1.0, color = "grey", zorder = -2)
					for ymajor in ax.yaxis.get_majorticklocs():
						ax.axhline(y = ymajor, ls = '--', alpha = 0.5, linewidth = 1.0, color = "grey", zorder = -2)


					# Legend
					leg = ax.legend(
						loc = ("upper left" if (args.num_clusters == 5 and clusterIdx != 4) else "upper right"),
						fontsize = LEGEND_FONT_SIZE + (2 if (clusterIdx in [0, 1]) else 0),
						labelspacing = 0.25,
						facecolor = "#FFFAE5", framealpha = 1.0)

					for lh in leg.legendHandles:
						lh.set_alpha(1)


					# Adjust x Limits
					x_limit_left, x_limit_right = ax.get_xlim()
					ax.set_xlim([x_limit_left + 0.175, x_limit_right - 0.175])


			plot_metric(plt, results, currMetric = metric)

			plt.tight_layout()
			plt.gcf().subplots_adjust(top = 0.94, left = 0.045, right = 0.99, wspace = 0.25, hspace = 0.3)
			mainfig.savefig(currFp)
			plt.close(mainfig)

			print("Bar Plot saved to \"{}\"..".format( currFp ))



			# Performance Grid (Table)
			currFp = resultsSummaryFp.replace(".txt", "{:s}_{:d}__table.png".format( metric, K ))

			print("Plotting Table for Test {:s}@{:d}...".format( metric, K ))
			colors = ["red", "deepskyblue", "yellow", "blueviolet", "peru", "orchid", "cyan", "greenyellow", "darkorange"]

			# Default fig size is [6.4, 4.8] (in inches, not pixels)
			mainfig = plt.figure(figsize = (20.0, 12.0), dpi = 300)

			plt.gcf().suptitle("Test {:s}@{:d}".format( metric, K ),
				fontsize = 20, fontweight = "bold", wrap = True, linespacing = 1.5)

			def plotTable(results, currMetric = "nDCG"):

				for clusterIdx, clusterDatasets in enumerate(lstClusterDatasets):

					for datasetIdx, clusterDataset in enumerate(clusterDatasets):

						if (clusterDataset not in results.keys()):
							continue

						# '# of datasets in cluster' rows, 'args.num_clusters' columns
						ax = plt.subplot(len(lstClusterDatasets[0]), args.num_clusters, (args.num_clusters * datasetIdx) + (clusterIdx + 1))

						datasetResults = results[clusterDataset]

						datasetName = clusterDataset.strip()
						# if (datasetName.startswith("Amazon (")):
						# 	datasetName = "Amazon\n{:s}".format( datasetName.replace("Amazon ", "") )

						lstScores = []
						for model in MODELS:
							lstScores.append(datasetResults[(model, K)] if (model, K) in datasetResults.keys() else 0.0)

						# Note: Negation, i.e. * -1
						lstRanks = rankdata([(x * -1) for x in lstScores], method = "min").astype(int)

						lstModelsRanks = []
						for model, rank in zip(MODELS, lstRanks):

							lstModelsRanks.append([
								model,
								rank
							])

						lstModelsRanks.sort(key = lambda x: x[1])
						lstRankedModels = [x[0] for x in lstModelsRanks]

						# print("Cluster {}, {}:".format( clusterIdx + 1, datasetName ))
						# for model in lstRankedModels:
						# 	print(model)

						modelColors = [to_rgba(colors[MODELS.index(m)], 0.5) for m in lstRankedModels]

						mvmTable = np.ndarray((len(lstRankedModels), len(lstRankedModels)), dtype = object)
						mvmTableColors = np.ndarray((len(lstRankedModels), len(lstRankedModels)), dtype = object)

						for mIdx1, m1 in enumerate(lstRankedModels):
							for mIdx2, m2 in enumerate(lstRankedModels):

								if (mIdx2 <= mIdx1):

									mvmTable[mIdx1][mIdx2] = "-"
									mvmTableColors[mIdx1][mIdx2] = to_rgba("dimgray", 0.5)

								else:
									
									m1Result = datasetResults[(m1, K)] if (m1, K) in datasetResults.keys() else 0.0
									m2Result = datasetResults[(m2, K)] if (m2, K) in datasetResults.keys() else 0.0

									percentImp = 0.0 if (m1Result == 0 or m2Result == 0) else (m1Result - m2Result) / m2Result * 100.0
									_, pValue = ttest_ind_from_stats(
										m2Result, m2Result, DATASET_2_DATA_POINTS[clusterDataset],
										m1Result, m1Result, DATASET_2_DATA_POINTS[clusterDataset])

									pValueSymbol = ""
									mvmTableColors[mIdx1][mIdx2] = "white"
									if (pValue < 0.05):
										pValueSymbol = " *"
										mvmTableColors[mIdx1][mIdx2] = to_rgba("palegreen", 0.5)
									if (pValue < 0.01):
										pValueSymbol = " **"
										mvmTableColors[mIdx1][mIdx2] = to_rgba("limegreen", 0.5)

									mvmTable[mIdx1][mIdx2] = "{:5.2f}%{}".format( percentImp, pValueSymbol )

						# print(mvmTable)

						ax.axis("off")
						t = ax.table(mvmTable, rowLabels = lstRankedModels, colLabels = lstRankedModels,
							rowColours = modelColors, colColours = modelColors,
							cellColours = mvmTableColors, cellLoc = "center",
							loc = "center", fontsize = 18)
						t.auto_set_font_size(False)
						t.set_fontsize(7)

						t.scale(1, 2.75)

						ax.text(0.5, 1.065, "Cluster {:d}, {}".format( clusterIdx + 1, datasetName ),
							fontsize = 10, fontweight = "bold", ha = "center", va = "bottom", ma = "center")

			plotTable(results, currMetric = metric)

			plt.tight_layout()
			plt.gcf().subplots_adjust(top = 0.9, left = 0.075, right = 0.95, wspace = 0.3, hspace = 0.325)
			mainfig.savefig(currFp)
			plt.close(mainfig)

			print("Table saved to \"{}\"..".format( currFp ))

	print("")
	exit()


