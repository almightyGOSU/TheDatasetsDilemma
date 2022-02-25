from util import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.ticker import LinearLocator, MultipleLocator, MaxNLocator, FormatStrFormatter, ScalarFormatter

from sklearn.metrics.pairwise import euclidean_distances



def plotFigure(arrValues, lstDatasets, metric = "Euclidean Distance"):

	mainfig = plt.figure(figsize = (22.5, 19.0), dpi = 300)

	plotTitle = "Dataset Similarities ({})".format( metric )
	plt.gcf().suptitle(plotTitle, fontsize = 32, fontweight = "bold", wrap = True, linespacing = 1.25)


	ax = plt.gca()
	im = ax.imshow(arrValues, cmap = "RdYlGn_r", aspect = "equal")

	# Create colorbar
	cbar = ax.figure.colorbar(im, ax = ax, pad = 0.02)

	cbar.ax.yaxis.set_major_locator(LinearLocator(numticks = 10))
	cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText = True))

	cbar.ax.tick_params(axis = "y", length = 8, labelsize = 20, pad = 4)

	offsetText = cbar.ax.yaxis.get_offset_text()
	offsetText.set_size(22)
	offsetText.set_style("italic")
	offsetText.set_ha("right")
	offsetText.set_va("bottom")
	offsetText.set_ma("left")

	cbar.ax.set_ylabel(metric, rotation = -90, va = "bottom", fontsize = 24, fontweight = "bold", labelpad = 15)


	ax.xaxis.set_major_locator(MaxNLocator(integer = True))
	ax.yaxis.set_major_locator(MaxNLocator(integer = True))

	ax.set_xticks(list(np.arange(0, len(lstDatasets))))
	ax.set_xticklabels([
		"$\\bf{(}$" + "{:d}".format( idx + 1 ) + "$\\bf{)}$" + " {:s}".format( dataset.replace("TPS", "Taste Profile Subset") )
		for idx, dataset in enumerate(lstDatasets)], fontsize = 16)

	ax.set_yticks(list(np.arange(0, len(lstDatasets))))
	ax.set_yticklabels([
		"$\\bf{(}$" + "{:d}".format( idx + 1 ) + "$\\bf{)}$" + " {:s}".format( dataset.replace("TPS", "Taste Profile Subset") )
		for idx, dataset in enumerate(lstDatasets)], fontsize = 16)

	# Rotate x-labels
	plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")


	plt.gcf().subplots_adjust(top = 0.94, left = 0.205, right = 0.985, bottom = 0.225)

	fp_image_file = "Dataset Similarities ({}).png".format( metric )
	mainfig.savefig(fp_image_file)
	plt.close(mainfig)

	print("\nDataset Similarities ({}) saved to: '{}'".format( metric, fp_image_file ))



if (__name__ == '__main__'):

	# Load dataset characteristics
	ALL_STATS_FILE = "../Datasets/characteristics_all.txt"
	datasetStats = loadStatistics(ALL_STATS_FILE)


	lstDatasets, embeddings = [], []
	for dataset, stats in sorted(datasetStats.items()):

		datasetName = dataset.strip().replace("Taste Profile Subset", "TPS")
		datasetName = datasetName.replace("Goodbooks", "Goodbooks-10k")

		# (0) numUsers, 		(1) numItems, 		(2) numInteractions, 	(3) density
		# (4) userMin, 			(5) userMax, 		(6) userAvg, 			(7) itemMin, 		(8) itemMax, 		(9) itemAvg
		# (10) spaceSizeLog, 	(11) shapeLog, 		(12) densityLog, 		(13) userGini, 		(14) itemGini

		# We are just using (10) to (14)
		datasetEmbedding = stats[10:]

		if (all(datasetEmbedding) == 0):

			print("'{}' does not have a valid embedding..".format( datasetName ))
			continue

		lstDatasets.append(datasetName)
		embeddings.append(datasetEmbedding)

	embeddings = np.array(embeddings)


	# Derive & plot pairwise euclidean distance
	euclideanDistances = euclidean_distances(embeddings)
	plotFigure(euclideanDistances, lstDatasets, metric = "Euclidean Distance")


	print("")
	exit()


