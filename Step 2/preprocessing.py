import argparse
import time

from util import *



if (__name__ == '__main__'):

	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input", 				dest = "input", 		type = str, 	default = "ML-100K.txt",
		help = "Input (Default: ML-100K.txt)")

	parser.add_argument("-p", "--partition", 			dest = "partition", 	type = int,  	default = 0,
		help = "Partition the dataset into train / validation / test (Default: 0, i.e. disabled)")

	parser.add_argument("-rs", "--random_seed", 		dest = "random_seed", 	type = int,  	default = 1337,
		help = "Random Seed (Default: 1337)")

	args = parser.parse_args()


	# Start timer
	startTime = time.time()

	# Source Folder (i.e. the original dataset)
	SOURCE_FOLDER = "../Datasets/Source/"

	datasetFilepath = "{}{}".format( SOURCE_FOLDER, args.input.strip() )
	dataset = args.input.strip().replace(".csv", "").replace(".txt", "").replace(".tsv", "").replace(".json", "")

	header, sep, names = getFileInfo(dataset)

	# Original Data
	rawDataOrig = readFile(dataset, datasetFilepath, header, sep, names)
	print("\nThe '{}' dataset has been loaded!".format( dataset ))


	# ================================================ Preprocessing =========================================================
	# ========================================================================================================================
	# Preprocess the dataset
	# Default: 5-core

	# 5-core (40 datasets)
	# All 24 Amazon datasets, Amazon Fine Food
	# BookCrossing, CiteULike-a, CiteULike-t, Epinions, FilmTrust, Flixster, GoodReads (Comics)
	# HetRec2011-LastFM-2K, Last.fm 1K, Last.fm 360K, Meetup (NYC), Netflix, Yahoo! R1, Yahoo! R4, Yelp
	if (dataset.startswith("Amazon") or
		dataset in ["BookCrossing", "CiteULike-a", "CiteULike-t", "Epinions", "FilmTrust", "Flixster", "GoodReads (Comics)"] or
		dataset in ["HetRec2011-LastFM-2K", "Last.fm 1K", "Last.fm 360K", "Meetup (NYC)", "Netflix", "Yahoo! R1", "Yahoo! R4", "Yelp"]):
		rawData, userActivity, itemPopularity = filter_triplets(rawDataOrig, min_uc = 5, min_ic = 5)

	# Retain users with at least 5 interactions (2 datasets)
	# Gowalla, Million Song Dataset
	elif (dataset in ["Gowalla", "Million Song Dataset"]):
		rawData, userActivity, itemPopularity = filter_triplets(rawDataOrig, min_uc = 5, min_ic = 0, recursive = False)

	# Retain items with at least 5 interactions (6 datasets)
	# HetRec2011-ML-2K, All 4 MovieLens datasets, Pinterest
	elif (dataset in ["HetRec2011-ML-2K", "Pinterest"] or dataset.startswith("ML-")):
		rawData, userActivity, itemPopularity = filter_triplets(rawDataOrig, min_uc = 0, min_ic = 5, recursive = False)

	# Do nothing (3 datasets)
	# Goodbooks, Million Song Dataset (Taste Profile Subset), Yahoo! R2
	elif (dataset in ["Goodbooks", "Million Song Dataset (Taste Profile Subset)", "Yahoo! R2"]):
		rawData, userActivity, itemPopularity = filter_triplets(rawDataOrig, min_uc = 0, min_ic = 0, recursive = False)

	# Do not include these datasets (3 datasets)
	# HetRec2011-Delicious-2K, Twitter (USA), Twitter (WW)
	elif (dataset in ["HetRec2011-Delicious-2K", "Twitter (USA)", "Twitter (WW)"]):
		print("[ERROR] Excluded dataset: {}".format( dataset ))
		exit()

	else:
		print("[ERROR] Unexpected dataset: {}".format( dataset ))
		exit()

	# Elapsed Time
	endTime = time.time()
	durationInSecs = endTime - startTime
	durationInMins = durationInSecs / 60
	print("\nDataset preprocessed after {:.2f} seconds ({:.2f} minutes)".format( durationInSecs, durationInMins ))
	# ========================================================================================================================


	# ================================================= Statistics ===========================================================
	# ========================================================================================================================
	# Get the dataset statistics
	datasetStats = deriveAllStatistics(rawData, userActivity, itemPopularity)

	# (0) numUsers, 		(1) numItems, 		(2) numInteractions, 	(3) density
	# (4) userMin, 			(5) userMax, 		(6) userAvg, 			(7) itemMin, 		(8) itemMax, 		(9) itemAvg
	# (10) spaceSizeLog, 	(11) shapeLog, 		(12) densityLog, 		(13) userGini, 		(14) itemGini

	# Output Folder (for storing statistics)
	OUTPUT_FOLDER = "../Datasets/Preprocessed/{}/".format( dataset )
	mkdir_p(OUTPUT_FOLDER)

	with codecs.open("{}{}".format( OUTPUT_FOLDER, "stats.txt"), "w", encoding = "utf-8", errors = "ignore") as outFile:
		outFile.write("{}, {}".format( dataset, ", ".join([str(x) for x in datasetStats]) ))

	# Elapsed Time
	endTime = time.time()
	durationInSecs = endTime - startTime
	durationInMins = durationInSecs / 60
	print("\nDataset statistics derived after {:.2f} seconds ({:.2f} minutes)".format( durationInSecs, durationInMins ))
	# ========================================================================================================================


	# ================================================ Partitioning ==========================================================
	# ========================================================================================================================
	if (args.partition):

		# Set random seed
		np.random.seed(args.random_seed)

		allUniqueUserIDs, allUniqueItemIDs = pd.unique(rawData["userID"]), pd.unique(rawData["itemID"])

		np.savetxt(os.path.join(OUTPUT_FOLDER, "users.txt"), allUniqueUserIDs, fmt = "%s")
		np.savetxt(os.path.join(OUTPUT_FOLDER, "items.txt"), allUniqueItemIDs, fmt = "%s")

		print("\n{:<15s} {:<18s} {:,d}".format( "[ALL]", "# of Users:", len(allUniqueUserIDs) ))
		print("{:<15s} {:<18s} {:,d}".format( "[ALL]", "# of Items:", len(allUniqueItemIDs) ))
		print("{:<15s} {:<18s} {:,d}".format( "[ALL]", "# of Interactions:", len(rawData) ))


		# By setting chrono = True, it performs 'Chronological Leave-One-Out' if there's timestamp, else 'Random Leave-One-Out'
		train, validation, test = leave_one_out(rawData, dataset, chrono = True)

		# Re-numbering the users and items
		user2uid = dict((uid, idx) for (idx, uid) in enumerate(allUniqueUserIDs))
		item2iid = dict((iid, idx) for (idx, iid) in enumerate(allUniqueItemIDs))

		def numerize(rawData, user2uid, item2iid):

			uid = list(map(lambda x: user2uid[x], rawData["userID"]))
			iid = list(map(lambda x: item2iid[x], rawData["itemID"]))
			return pd.DataFrame(data = {"userID": uid, "itemID": iid}, columns = ["userID", "itemID"])

		train = numerize(train, user2uid, item2iid)
		validation = numerize(validation, user2uid, item2iid)
		test = numerize(test, user2uid, item2iid)

		print("\n{:<15s} {:<18s} {:,d}".format( "[Training]", "# of Interactions:", len(train) ))
		print("{:<15s} {:<18s} {:,d}".format( "[Validation]", "# of Interactions:", len(validation) ))
		print("{:<15s} {:<18s} {:,d}".format( "[Testing]", "# of Interactions:", len(test) ))

		trainUsers, trainItems = set(pd.unique(train["userID"])), set(pd.unique(train["itemID"]))
		validationUsers, validationItems = set(pd.unique(validation["userID"])), set(pd.unique(validation["itemID"]))
		testUsers, testItems = set(pd.unique(test["userID"])), set(pd.unique(test["itemID"]))

		assert (validationUsers <= trainUsers and testUsers <= trainUsers), "'validation' or 'testing' users not in 'training'!" 


		# Saving train, validation and test sets
		trainFp = os.path.join(OUTPUT_FOLDER, "train.csv")
		train.to_csv(trainFp, index = False)
		print("\nTraining Set saved to '{}'..".format( trainFp ))

		validationFp = os.path.join(OUTPUT_FOLDER, "validation.csv")
		validation.to_csv(validationFp, index = False)
		print("Validation Set saved to '{}'..".format( validationFp ))

		testFp = os.path.join(OUTPUT_FOLDER, "test.csv")
		test.to_csv(testFp, index = False)
		print("Testing Set saved to '{}'..".format( testFp ))

		# Elapsed Time
		endTime = time.time()
		durationInSecs = endTime - startTime
		durationInMins = durationInSecs / 60
		print("\nDataset partitioned after {:.2f} seconds ({:.2f} minutes)".format( durationInSecs, durationInMins ))
	# ========================================================================================================================

	print("")
	exit()


