from util import *



if (__name__ == '__main__'):

	# Source Folder
	SOURCE_FOLDER = "../Datasets/Preprocessed/"

	# Datasets
	lstDatasets = sorted([os.path.join(SOURCE_FOLDER, f) for f in os.listdir(SOURCE_FOLDER)])


	# ======================================= Gather all the statistics ======================================================
	# ========================================================================================================================
	datasetStats = defaultdict(list)
	for datasetFilepath in lstDatasets:

		dataset = datasetFilepath.strip().split("/")[-1].strip()

		statsFilepath = "{}/{}".format( datasetFilepath, "stats.txt" )
		if (os.path.exists(statsFilepath)):

			with codecs.open(statsFilepath, "r", encoding = "utf-8", errors = "ignore") as inFile:
				line = inFile.readline()

			elements = line.strip().split(", ")

			# (0) numUsers, 		(1) numItems, 		(2) numInteractions, 	(3) density
			# (4) userMin, 			(5) userMax, 		(6) userAvg, 			(7) itemMin, 		(8) itemMax, 		(9) itemAvg
			# (10) spaceSizeLog, 	(11) shapeLog, 		(12) densityLog, 		(13) userGini, 		(14) itemGini

			stats = [x.strip() for x in elements[1:]]
			stats = [int(float(x)) if int(float(x)) == float(x) else float(x) for x in stats]

			datasetStats[dataset] = stats

	ALL_STATS_FILE = "../Datasets/characteristics_all.txt"

	with codecs.open(ALL_STATS_FILE, "w", encoding = "utf-8", errors = "ignore") as outFile:
		for dataset, stats in sorted(datasetStats.items()):
			outFile.write("{}, {}\n".format( dataset, ", ".join([str(x) for x in stats]) ))
	# ========================================================================================================================


	# ========================================= Generate some tables =========================================================
	# ========================================================================================================================
	# Not necessary at all, just to demonstrate the function
	ALL_STATS_FILE = "../Datasets/characteristics_all.txt"
	datasetStats = loadStatistics(ALL_STATS_FILE)

	generateTables(datasetStats)
	# ========================================================================================================================

	print("")
	exit()


