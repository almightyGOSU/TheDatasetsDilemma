from util import *

import argparse
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.ticker import LinearLocator, MultipleLocator, MaxNLocator, FormatStrFormatter, ScalarFormatter
from matplotlib.patches import Polygon

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

from scipy.spatial import ConvexHull



# Generates a mapping from each cluster to its set of datasets
def mapDatasetsToClusters(lstDatasets, clusterLabels):

	clusterToDatasets = defaultdict(list)
	for c, i in enumerate(clusterLabels):
		clusterToDatasets[i].append(lstDatasets[c])

	return clusterToDatasets


# Plot datasets using t-SNE along with the clustering information
def plotTSNE(lstDatasets, embeddings2D, clusterToDatasets, datasets_numPapers, metric = "Euclidean Distance", numClusters = 5):

	colors = ["red", "limegreen", "deepskyblue", "darkorange", "blueviolet", "peru", "orchid", "cyan", "greenyellow", "yellow"]

	# Circle, Square, Pentagon, Star, Plus, Triangle Up, Triangle Down, Diamond, X, Triangle Left, Triangle Right
	# Club, Heart, Spade, Musical Note [15 symbols]
	markers = ["o", "s", "p", "*", "P", "^", "v", "D", "X", "<", ">",
				"$\\clubsuit$", "$\u2665$", "$\\spadesuit$", "$\u266B$"]

	enlargedMarkers = ["*", "$\\clubsuit$", "$\u2665$", "$\\spadesuit$", "$\u266B$"]


	mainfig = plt.figure(figsize = (25.0, 18.0), dpi = 300)

	plotTitle = "Datasets & Clustering Visualisation with t-SNE ({} Clusters)".format( numClusters )
	plt.gcf().suptitle(plotTitle, fontsize = 32, fontweight = "bold", wrap = True, linespacing = 1.25)


	lstDatasetsOrdered, lstEmbeddingsOrdered, lstClusterIdx = [], [], []
	for clusterIdx, clusterDatasets in sorted(clusterToDatasets.items()):

		for dataset in sorted(clusterDatasets):

			lstDatasetsOrdered.append(dataset)
			lstEmbeddingsOrdered.append(embeddings2D[lstDatasets.index(dataset)])
			lstClusterIdx.append(clusterIdx)


	currClusterIdx, currMarkerIdx = 0, 0
	bMarkerEdge = False
	bFillTop = False

	lstClusterPoints, lstPoints = [], []
	for dataset, embedding2D, clusterIdx in zip(lstDatasetsOrdered, lstEmbeddingsOrdered, lstClusterIdx):

		if (currClusterIdx != clusterIdx):

			if (lstPoints):
				lstClusterPoints.append(lstPoints)
				lstPoints = []

			currClusterIdx = clusterIdx
			currMarkerIdx = 0
			bMarkerEdge = False
			bFillTop = False

		lstPoints.append(embedding2D)
		plt.plot(embedding2D[0], embedding2D[1], color = colors[clusterIdx % len(colors)], label = dataset,
			marker = markers[currMarkerIdx],
			markersize = 28 if (markers[currMarkerIdx] in enlargedMarkers) else 20,
			markeredgecolor = "black" if bMarkerEdge else colors[clusterIdx % len(colors)],
			markeredgewidth = 2 if bMarkerEdge else 0,
			fillstyle = "top" if bFillTop else "full",
			linestyle = "None")


		if (dataset in datasets_numPapers.keys()):

			# Coloring scheme based on usage frequency
			# >= 5 papers 				---> Green
			# > 1 paper and < 5 papers 	---> Blue
			# 1 paper 					---> Red

			numPapers = datasets_numPapers[dataset]
			plt.plot(embedding2D[0], embedding2D[1],
				"go" if numPapers >= 5 else ("ro" if numPapers == 1 else "bo"),
				markersize = 100 + min(40, (numPapers - 5) * 3) if numPapers >= 5 else (60 if numPapers == 1 else 80),
				alpha = 0.075,
				zorder = -1,
				markeredgecolor = "white",
				markeredgewidth = 0)

			plt.annotate(text = "{:<2d}".format( numPapers ), xy = (embedding2D[0], embedding2D[1]),
				textcoords = "offset points", xytext = (15, 15),
				ha = "left", va = "bottom",
				fontsize = 24,
				fontweight = "extra bold" if numPapers >= 5 else ("normal" if numPapers == 1 else "bold"),
				fontstyle = "italic" if numPapers >= 5 else "normal",
				color = "green" if numPapers >= 5 else ("red" if numPapers == 1 else "blue"))

		currMarkerIdx += 1
		if (currMarkerIdx >= len(markers)):

			if (bMarkerEdge):
				bFillTop = True

			currMarkerIdx = 0
			bMarkerEdge = True


	# For the last cluster
	lstClusterPoints.append(lstPoints)

	left, right = plt.xlim()
	bottom, top = plt.ylim()

	# Convex Hull for each cluster
	for clusterIdx, lstPoints in enumerate(lstClusterPoints):

		if (len(lstPoints) <= 2):
			continue

		hull = ConvexHull(lstPoints)

		cent = np.mean(lstPoints, 0)
		lstAdjustedPoints = []
		for simplex in hull.simplices:
			lstAdjustedPoints.append(lstPoints[simplex[0]])
			lstAdjustedPoints.append(lstPoints[simplex[1]])

		lstAdjustedPoints.sort(key = lambda p: np.arctan2(p[1] - cent[1], p[0] - cent[0]))
		lstAdjustedPoints = lstAdjustedPoints[0::2]  # Deleting duplicates
		lstAdjustedPoints.insert(len(lstAdjustedPoints), lstAdjustedPoints[0])

		k = 1.3
		color = colors[clusterIdx % len(colors)]

		poly = Polygon(k * (np.array(lstAdjustedPoints) - cent) + cent,
			edgecolor = color, facecolor = color,
			fill = False, linewidth = 1.75, ls = "--",
			alpha = 0.5, zorder = -1.75)

		poly.set_capstyle("round")
		plt.gca().add_patch(poly)

	ax = plt.gca()
	ax.set_xlim(left, right)
	ax.set_ylim(bottom, top)


	pltLegend = plt.legend(fontsize = 15, ncol = 1, loc = "center left", bbox_to_anchor = (1.01, 0.5),
		title = ("$\\bf{Datasets}$" + " (by Cluster)"), title_fontsize = 24)

	# Resize labels in the legend
	for handle in pltLegend.legendHandles:
		marker = handle._legmarker.get_marker()
		handle._legmarker.set_markersize(16 if (marker in enlargedMarkers) else 13)
		handle._legmarker.set_markeredgewidth(1.4)


	ax = plt.gca()
	ax.tick_params(axis = "both", length = 6, labelsize = 20)

	left, right = plt.xlim()
	bottom, top = plt.ylim()

	# Grid
	for xmajor in ax.xaxis.get_majorticklocs():
		ax.axvline(x = xmajor, ls = '--', alpha = 0.5, linewidth = 1.0, color = "grey", zorder = -2)
	for ymajor in ax.yaxis.get_majorticklocs():
		ax.axhline(y = ymajor, ls = '--', alpha = 0.5, linewidth = 1.0, color = "grey", zorder = -2)

	ax.set_xlim(left, right)
	ax.set_ylim(bottom, top)


	plt.gcf().subplots_adjust(top = 0.915, left = 0.05, right = 0.775, bottom = 0.05)

	fp_image_file = "Clustering Visualisation ({} Clusters).png".format( numClusters )
	mainfig.savefig(fp_image_file)
	plt.close(mainfig)

	print("\nClustering Visualisation ({} Clusters) saved to: '{}'".format( numClusters, fp_image_file ))



if (__name__ == '__main__'):

	parser = argparse.ArgumentParser()

	parser.add_argument("-nc", 		"--num_clusters", 		dest = "num_clusters", 		type = int, 	default = 5,
		help = "Number of clusters (Default: 5)")

	parser.add_argument("-iter", 	"--iterations", 		dest = "iterations", 		type = int,  	default = 100,
		help = "Number of runs for k-means (Default: 100)")

	parser.add_argument("-rs", 		"--random_seed", 		dest = "random_seed", 		type = int,  	default = 1337,
		help = "Random Seed (Default: 1337)")

	args = parser.parse_args()


	startTime = time.time()


	# ======================================== Load dataset statistics =======================================================
	# ========================================================================================================================
	# Load dataset characteristics
	ALL_STATS_FILE = "../Datasets/characteristics_all.txt"
	datasetStats = loadStatistics(ALL_STATS_FILE)

	lstDatasets, embeddings = [], []
	for dataset, stats in sorted(datasetStats.items()):

		datasetName = dataset.strip()

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

	print("\n# of Datasets: {}".format( len(lstDatasets) ))
	print("Embeddings Matrix: {}".format( embeddings.shape ))
	# ========================================================================================================================


	# ======================================== Clustering the datasets =======================================================
	# ========================================================================================================================
	# n_init: 	Number of time the k-means algorithm will be run with different centroid seeds
	#			The final results will be the best output of 'n_init' consecutive runs in terms of inertia
	kmeans_model = KMeans(init = 'k-means++',
		n_clusters = args.num_clusters,
		n_init = args.iterations,
		random_state = args.random_seed)

	print("\nPerforming K-Means clustering to obtain {} clusters after a total of {} runs..".format(
		args.num_clusters, args.iterations ))

	kmeans_model.fit(embeddings)

	clusterLabels = kmeans_model.labels_
	clusterToDatasets = mapDatasetsToClusters(lstDatasets, clusterLabels)
	clusterCenters = kmeans_model.cluster_centers_

	# Elapsed Time
	endTime = time.time()
	durationInSecs = endTime - startTime
	durationInMins = durationInSecs / 60
	print("K-means clustering completed after {:.2f} seconds ({:.2f} minutes)".format( durationInSecs, durationInMins ))
	# ========================================================================================================================


	# ===================================== Store the clustering results =====================================================
	# ========================================================================================================================

	# Super detailed version
	# This displays ALL Statistics for ALL Datasets in each Cluster
	outFp = "./Clustering ({} Clusters) (Detailed).txt".format( args.num_clusters )
	with codecs.open(outFp, "w", encoding = "utf-8", errors = "ignore") as outFile:

		headerStr = ("|| {:<35s}" + " || {:^11s}" + " | {:^11s}" + " | {:^14s}" + " | {:^9s} ||" + 
				" {:^12s}" + " | {:^10s}" + " | {:^12s}" + " | {:^10s}" + " | {:^10s} ||\n")
		fmtStr1 = ("|| {:<35s}" + " || {:>11,.0f}" + " | {:>11,.0f}" + " | {:>14,.0f}" + " | {:>9.6f} ||" + 
					" {:>12.3f}" + " | {:>10.3f}" + " | {:>12.3f}" + " | {:>10.3f}" + " | {:>10.3f} ||\n")
		fmtStr2 = ("|| {:<35s}" + " || {:>11,.2f}" + " | {:>11,.2f}" + " | {:>14,.2f}" + " | {:>9.6f} ||" + 
					" {:>12.3f}" + " | {:>10.3f}" + " | {:>12.3f}" + " | {:>10.3f}" + " | {:>10.3f} ||\n")
		linebreak = "||" + "-" * 37 + "||" + "-" * 56 + "||" + "-" * 68 + "||\n"

		for clusterIdx, clusterDatasets in sorted(clusterToDatasets.items()):

			lstClusterStats = []
			for clusterDataset in clusterDatasets:

				stats = datasetStats[clusterDataset]
				lstClusterStats.append([stats[:4] + stats[10:]])

			npClusterStats = np.concatenate(lstClusterStats)

			minClusterStats = list(np.min(npClusterStats, axis = 0))
			maxClusterStats = list(np.max(npClusterStats, axis = 0))
			avgClusterStats = list(np.mean(npClusterStats, axis = 0))

			outFile.write(linebreak)
			outFile.write(headerStr.format( "Cluster {} Summary Statistics".format( clusterIdx + 1),
				"Users", "Items", "Interactions", "Density",
				"spaceSizeLog", "shapeLog", "densityLog", "userGini", "itemGini" ))
			outFile.write(linebreak)

			outFile.write(fmtStr1.format( *(["Minimum"] + minClusterStats) ))
			outFile.write(fmtStr1.format( *(["Maximum"] + maxClusterStats) ))
			outFile.write(fmtStr2.format( *(["Average"] + avgClusterStats) ))
			outFile.write(linebreak)

			outFile.write(headerStr.format( "Cluster {} Datasets".format( clusterIdx + 1),
				"Users", "Items", "Interactions", "Density",
				"spaceSizeLog", "shapeLog", "densityLog", "userGini", "itemGini" ))
			outFile.write(linebreak)

			for clusterDataset in sorted(clusterDatasets):

				datasetName = clusterDataset.replace("(Taste Profile Subset)", "(TPS)")
				stats = datasetStats[clusterDataset]
				outFile.write(fmtStr1.format( *([datasetName] + stats[:4] + stats[10:]) ))

			outFile.write(linebreak)
			outFile.write("\n")

	# Simple version
	# Only shows the cluster centroid, and lists the datasets in each cluster
	outFp = "./Clustering ({} Clusters) (Simple).txt".format( args.num_clusters )
	with codecs.open(outFp, "w", encoding = "utf-8", errors = "ignore") as outFile:

		headerStr = ("|| {:<15s} || {:^12s} | {:^12s} | {:^12s} | {:^12s} | {:^12s} ||\n")
		headerStr = headerStr.format( "", "spaceSizeLog", "shapeLog", "densityLog", "userGini", "itemGini" )
		linebreak = "||" + "-" * 17 + "||" + "-" * 74 + "||\n"
		statsFmtStr = ("|| {:<15s} || {:>12.3f} | {:>12.3f} | {:>12.3f} | {:>12.3f} | {:>12.3f} ||\n")

		for clusterIdx, clusterDatasets in sorted(clusterToDatasets.items()):

			outFile.write("\n{}{}{}{}\n".format( linebreak, headerStr, statsFmtStr.format(
				"Cluster {}".format( clusterIdx + 1 ), *clusterCenters[clusterIdx] ), linebreak ))
			outFile.write("Cluster {} [{} datasets]:\n".format( clusterIdx + 1, len(clusterDatasets) ))

			for idx, clusterDataset in enumerate(sorted(clusterDatasets)):
				outFile.write("{:<4s} {:<s}\n".format( "({:d})".format( idx + 1 ), clusterDataset ))

			outFile.write("\n")

		outFile.write("\n")
	# ========================================================================================================================


	# =================================== Visualise datasets and clustering ==================================================
	# ========================================================================================================================
	# Datasets used in the 48 papers, and the corresponding papers for each dataset
	datasets_papers = getDatasetsPapers("../Step 1/datasets_to_papers.txt")
	datasets_numPapers = {dataset: len(papers) for dataset, papers in datasets_papers.items()}

	# t-SNE
	tSNE = TSNE(n_components = 2, random_state = args.random_seed, metric = "euclidean", method = "exact")
	embeddings2D = tSNE.fit_transform(embeddings)
	plotTSNE(lstDatasets, embeddings2D, clusterToDatasets, datasets_numPapers, numClusters = args.num_clusters)

	# Elapsed Time
	endTime = time.time()
	durationInSecs = endTime - startTime
	durationInMins = durationInSecs / 60
	print("Clustering Visualisation completed after {:.2f} seconds ({:.2f} minutes)".format( durationInSecs, durationInMins ))
	# ========================================================================================================================


	# ====================================== Sample datasets for Step 3 ======================================================
	# ========================================================================================================================
	NUM_REP = 3

	clusterToRepDatasets = defaultdict(list)
	for clusterIdx, clusterDatasets in sorted(clusterToDatasets.items()):

		# No sampling required if the number of datasets in the cluster is not more than the number of 'representative' datasets
		if (len(clusterDatasets) <= NUM_REP):

			clusterToRepDatasets[clusterIdx] = clusterDatasets
			continue

		# Closest
		else:

			clusterCenter = clusterCenters[clusterIdx]
			clusterDatasetEmbeddings = []

			for dataset in clusterDatasets:
				clusterDatasetEmbeddings.append(embeddings[lstDatasets.index(dataset)])

			distances = np.squeeze(euclidean_distances([clusterCenter], clusterDatasetEmbeddings))
			distances_temp = np.copy(distances)

			for _ in range(NUM_REP):

				datasetIdx = np.argmin(distances_temp)
				clusterToRepDatasets[clusterIdx].append(clusterDatasets[datasetIdx])
				distances_temp[datasetIdx] = np.inf


	outFp = "./Sampled Datasets ({} Clusters).txt".format( args.num_clusters )
	with codecs.open(outFp, "w", encoding = "utf-8", errors = "ignore") as outFile:

		for clusterIdx, clusterRepDatasets in sorted(clusterToRepDatasets.items()):

			numOriginalDatasets = len(clusterToDatasets[clusterIdx])

			outFile.write("Cluster {} [{} (out of {:d}) 'representative' datasets]:\n".format(
				clusterIdx + 1, len(clusterRepDatasets), numOriginalDatasets ))

			for idx, clusterDataset in enumerate(sorted(clusterRepDatasets)):
				outFile.write("{:<4s} {:<s}\n".format( "({:d})".format( idx + 1 ), clusterDataset ))

			outFile.write("\n\n")
	# ========================================================================================================================

	print("")
	exit()


