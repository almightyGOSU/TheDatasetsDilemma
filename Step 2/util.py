from collections import defaultdict
from tqdm import tqdm

import codecs
import math
import numpy as np
import os
import pandas as pd



# These datasets are chronologically ordered, even though they do not have a timestamp column
CHRONO_DATASETS = ["Goodbooks"]


# Creating a folder
def mkdir_p(path):

	if (path == ""):
		return
	try:
		os.makedirs(path)
	except:
		pass


def getFileInfo(dataset):

	header, sep, names = None, None, None
	dataset = dataset.strip()

	# All 24 Amazon Datasets from 'jmcauley.ucsd.edu/data/amazon'
	if (dataset.startswith("Amazon (")):
		sep, names = ",", ["userID", "itemID", "rating", "timestamp"]

	# Amazon Fine Food from 'kaggle.com/snap/amazon-fine-food-reviews'
	elif (dataset == "Amazon Fine Food"):
		header, sep, names = 0, ",", ["userID", "itemID", "rating", "timestamp"]

	# BookCrossing (No timestamps)
	# Contains 278,858 users providing 1,149,780 ratings (explicit / implicit) about 271,379 books
	elif (dataset == "BookCrossing"):
		header, sep, names = 0, ";", ["userID", "itemID", "rating"]

	# CiteULike-a & CiteULike-t (No timestamps)
	elif (dataset in ["CiteULike-a", "CiteULike-t"]):
		names = ["userID", "itemID", "rating"]

	# Epinions (No timestamps)
	elif (dataset == "Epinions"):
		sep, names = " ", ["userID", "itemID", "rating"]

	# FilmTrust (No timestamps)
	elif (dataset == "FilmTrust"):
		sep, names = " ", ["userID", "itemID", "rating"]

	# Flixster
	elif (dataset == "Flixster"):
		header, sep, names = 0, "\t", ["userID", "itemID", "rating", "timestamp"]

	# Goodbooks (No timestamps, BUT the ratings are already sorted by time)
	elif (dataset == "Goodbooks"):
		header, sep, names = 0, ",", ["userID", "itemID", "rating"]

	# GoodReads (Comics) (No timestamps)
	elif (dataset == "GoodReads (Comics)"):
		sep, names = ",", ["userID", "itemID", "rating"]

	# Gowalla (No timestamps)
	# We are using the subset whereby..
	# - Venues with at least 20 check-ins are retained
	elif (dataset == "Gowalla"):
		sep, names = "\t", ["userID", "itemID", "rating"]

	# HetRec2011-Delicious-2K
	elif (dataset == "HetRec2011-Delicious-2K"):
		header, sep, names = 0, "\t", ["userID", "itemID", "rating", "timestamp"]

	# HetRec2011-LastFM-2K (No timestamps)
	elif (dataset == "HetRec2011-LastFM-2K"):
		header, sep, names = 0, "\t", ["userID", "itemID", "rating"]

	# HetRec2011-ML-2K
	elif (dataset == "HetRec2011-ML-2K"):
		header, sep, names = 0, "\t", ["userID", "itemID", "rating", "timestamp"]

	# Last.fm 1K
	elif (dataset == "Last.fm 1K"):
		sep, names = "\t", ["userID", "itemID", "rating", "timestamp"]

	# Last.fm 360K (No timestamps)
	elif (dataset == "Last.fm 360K"):
		sep, names = ",", ["userID", "itemID", "rating"]

	# Meetup (NYC) (No timestamps)
	elif (dataset == "Meetup (NYC)"):
		names = ["userID", "itemID", "rating"]

	# Million Song Dataset (No timestamps)
	# We are using the subset whereby..
	# - Users with at least 20 songs in their listening history, and
	# - Songs that are listened to by at least 200 users
	# are retained, since ALL 5 papers follow this approach
	elif (dataset == "Million Song Dataset"):
		sep, names = "\t", ["userID", "itemID", "rating"]

	# Million Song Dataset (Taste Profile Subset) (No timestamps)
	# We are using the subset whereby..
	# - Users with at least 20 songs in their listening history, and
	# - Songs that are listened to by at least 200 users are retained
	elif (dataset == "Million Song Dataset (Taste Profile Subset)"):
		sep, names = "\t", ["userID", "itemID", "rating"]

	# ML-100K
	elif (dataset == "ML-100K"):
		sep, names = "\t", ["userID", "itemID", "rating", "timestamp"]

	# ML-1M, ML-10M
	elif (dataset in ["ML-1M", "ML-10M"]):
		sep, names = "::", ["userID", "itemID", "rating", "timestamp"]

	# ML-20M
	elif (dataset == "ML-20M"):
		header, sep, names = 0, ",", ["userID", "itemID", "rating", "timestamp"]

	# Netflix (No timestamps)
	elif (dataset == "Netflix"):
		sep, names = ",", ["userID", "itemID", "rating"]

	# Pinterest (No timestamps)
	# We are using the preprocessed version whereby..
	# - Users with at least 20 interactions (pins) are retained
	elif (dataset == "Pinterest"):
		sep, names = "\t", ["userID", "itemID", "rating"]

	# Twitter (USA)
	elif (dataset == "Twitter (USA)"):
		sep, names = "\t", ["userID", "itemID", "rating", "timestamp"]

	# Twitter (WW)
	elif (dataset == "Twitter (WW)"):
		sep, names = "\t", ["userID", "itemID", "rating", "timestamp"]

	# Yahoo! R1 (No timestamps)
	elif (dataset == "Yahoo! R1"):
		sep, names = "\t", ["userID", "itemID", "rating"]

	# Yahoo! R2 (No timestamps)
	# The data has been trimmed so that each user has rated at least 20 songs, and each song has been rated by at least 20 users.
	# The data has been randomly partitioned into 10 equally sized sets of users to enable cross-validation techniques.
	# The two papers here are following FISM (KDD 2013), which randomly sampled a very small subset of users and items.
	# Instead of following these papers, we are directly using the first subset of users.
	elif (dataset == "Yahoo! R2"):
		sep, names = "\t", ["userID", "itemID", "rating"]

	# Yahoo! R4 (No timestamps)
	elif (dataset == "Yahoo! R4"):
		sep, names = "\t", ["userID", "itemID", "rating"]

	# Yelp
	elif (dataset == "Yelp"):
		sep, names = ",", ["userID", "itemID", "rating", "timestamp"]

	return header, sep, names


def readFile(dataset, datasetFilepath, header, sep, names):

	rawData = None

	# For CiteULike-a & CiteULike-t (No timestamps)
	# The user-item interaction is stored as a list of lists
	# Each (sub-)list contains the items consumed by each user (i.e. the cited articles)
	if (dataset in ["CiteULike-a", "CiteULike-t"]):

		with codecs.open(datasetFilepath, "r", encoding = "utf-8", errors = "ignore") as inFile:
			contents = inFile.readlines()

		rawData = []
		for userIdx, items in enumerate(contents):
			for itemIdx in sorted(items.split(" ")):
				rawData.append([userIdx, itemIdx, 1])

		rawData = pd.DataFrame(rawData, columns = names)

	# Flixster (No timestamps)
	elif (dataset == "Flixster"):

		rawData = pd.read_csv(datasetFilepath, header = header, sep = sep, names = names, encoding = "utf-16")

	# HetRec2011-Delicious-2K
	elif (dataset == "HetRec2011-Delicious-2K"):

		rawData = pd.read_csv(datasetFilepath, header = header, sep = sep, names = names, engine = "python")
		rawData = rawData.sort_values("timestamp").drop_duplicates(["userID", "itemID"], keep = "last")

	# Last.fm 360K (No timestamps)
	elif (dataset == "Last.fm 360K"):

		# Use C engine (It's faster)
		rawData = pd.read_csv(datasetFilepath, header = header, sep = sep, names = names)

	# For Meetup (NYC) (No timestamps)
	# The user-item interaction is stored as a list of lists
	# Each (sub-)list contains the users at each event
	elif (dataset == "Meetup (NYC)"):

		with codecs.open(datasetFilepath, "r", encoding = "utf-8", errors = "ignore") as inFile:
			contents = inFile.readlines()

		rawData = []
		for event_users in contents:

			event_users = event_users.strip().split()
			eventIdx = event_users[0]

			for userIdx in sorted(event_users[1:]):
				rawData.append([userIdx, eventIdx, 1])

		rawData = pd.DataFrame(rawData, columns = names)

	else:

		rawData = pd.read_csv(datasetFilepath, header = header, sep = sep, names = names, engine = "python")

	return rawData


def filter_triplets(tp, min_uc = 1, min_ic = 1, recursive = True):
	"""
	Filter out less active users and less active items (iteratively; to get a k-core)

	Input:
	tp (DataFrame): 			userID, itemID, rating, timestamp

	Output:
	tp (DataFrame): 			Filtered records
	currUserCount (Series): 	Number of items consumed by each user
	currItemCount (Series): 	Number of users who interacted with each item
	"""
	def get_count(df, id):
		return df.groupby(id).size()

	while True:

		oldUserCount, oldItemCount = get_count(tp, "userID"), get_count(tp, "itemID")

		# Remove less active users
		if (min_uc > 0):
			currUserCount = get_count(tp, "userID")
			tp = tp[tp["userID"].isin(currUserCount.index[currUserCount >= min_uc])]

		# Remove less active items
		if (min_ic > 0):
			currItemCount = get_count(tp, "itemID")
			tp = tp[tp["itemID"].isin(currItemCount.index[currItemCount >= min_ic])]

		currUserCount, currItemCount = get_count(tp, "userID"), get_count(tp, "itemID")

		if (not recursive or (oldUserCount.shape[0] == currUserCount.shape[0] and oldItemCount.shape[0] == currItemCount.shape[0])):
			break

	return tp, currUserCount, currItemCount


def basic_statistics(rawData, userActivity, itemPopularity):

	density = 0
	numUsers, numItems = userActivity.shape[0], itemPopularity.shape[0]
	numInteractions = rawData.shape[0]

	if (numUsers and numItems):
		density = (numInteractions / (numUsers * numItems)) * 100

	return numUsers, numItems, numInteractions, density


def detailed_statistics(userActivity, itemPopularity):

	userMin, userMax, userAvg = 0, 0, 0.0
	itemMin, itemMax, itemAvg = 0, 0, 0.0

	if (userActivity.shape[0]):
		userMin, userMax, userAvg = userActivity.min(), userActivity.max(), userActivity.mean()

	if (itemPopularity.shape[0]):
		itemMin, itemMax, itemAvg = itemPopularity.min(), itemPopularity.max(), itemPopularity.mean()

	return userMin, userMax, userAvg, itemMin, itemMax, itemAvg


# For calculating the gini coefficient over a sorted discrete frequency distribution
# - A value of 0 represents total equality (all items are equally popular)
# - A value of 1 represents maximal inequality (one bestselling item has all the interactions/ratings)
def gini_coeff(frequencyDistribution):

	numSamples = frequencyDistribution.shape[0]

	numInteractions = np.sum(frequencyDistribution)

	if (numInteractions == 0):
		return 0

	indices = np.arange(1, numSamples + 1, 1)
	indices = ((numSamples + 1) - indices) / (numSamples + 1)

	giniCoeff = 1 - 2 * np.sum((indices * frequencyDistribution) / numInteractions)

	return giniCoeff


def advanced_statistics(rawData, userActivity, itemPopularity):

	U, I, K = userActivity.shape[0], itemPopularity.shape[0], rawData.shape[0]

	if (not U or not I or not K):
		return 0.0, 0.0, 0.0, 0.0, 0.0

	spaceSizeLog = math.log10((U * I) / 1000)
	shapeLog = math.log10(U / I)
	densityLog = math.log10(K / (U * I))

	userGini = gini_coeff(userActivity.sort_values().values)
	itemGini = gini_coeff(itemPopularity.sort_values().values)

	return spaceSizeLog, shapeLog, densityLog, userGini, itemGini


# Basic, Detailed, and Advanced Statistics
def deriveAllStatistics(rawData, userActivity, itemPopularity):

	numUsers, numItems, numInteractions, density = basic_statistics(rawData, userActivity, itemPopularity)
	userMin, userMax, userAvg, itemMin, itemMax, itemAvg = detailed_statistics(userActivity, itemPopularity)
	spaceSizeLog, shapeLog, densityLog, userGini, itemGini = advanced_statistics(rawData, userActivity, itemPopularity)

	# (0) numUsers, 		(1) numItems, 		(2) numInteractions, 	(3) density
	# (4) userMin, 			(5) userMax, 		(6) userAvg, 			(7) itemMin, 		(8) itemMax, 		(9) itemAvg
	# (10) spaceSizeLog, 	(11) shapeLog, 		(12) densityLog, 		(13) userGini, 		(14) itemGini

	return [numUsers, numItems, numInteractions, density,
			userMin, userMax, userAvg, itemMin, itemMax, itemAvg,
			spaceSizeLog, shapeLog, densityLog, userGini, itemGini]


def leave_one_out(rawData, dataset, chrono = True):

	dataGroupedByUser = rawData.groupby("userID")
	train, validation, test = [], [], []

	print("")
	for _, group in tqdm(dataGroupedByUser, desc = "Partitioning '{}'".format( dataset )):

		# The number of items consumed by one user
		numInteractions = len(group)

		if (numInteractions >= 5):

			# Chronological Leave-One-Out (for datasets w/ timestamps)
			if (chrono and "timestamp" in rawData.columns):
				group = group.sort_values("timestamp")[["userID", "itemID", "rating"]]

			# Chronological Leave-One-Out (for datasets which are already chronologically ordered)
			elif (chrono and dataset in CHRONO_DATASETS):
				# Do nothing
				pass

			# Random Leave-One-Out
			else:
				group = group.sample(frac = 1).reset_index(drop = True)

			# Only keeping the 'userID' & 'itemID'
			train.append(group.iloc[:-2, :2])
			validation.append(group.iloc[-2, :2])
			test.append(group.iloc[-1, :2])

		# For users w/ less than 5 interactions
		else:
			train.append(group)

	train = pd.concat(train)
	validation = pd.DataFrame(validation, columns = ["userID", "itemID"])
	test = pd.DataFrame(test, columns = ["userID", "itemID"])

	return train, validation, test


# Loads all dataset characteristics
def loadStatistics(allStatsFp):

	if (os.path.exists(allStatsFp)):

		with codecs.open(allStatsFp, "r", encoding = "utf-8", errors = "ignore") as inFile:
			contents = inFile.readlines()

		datasetStats = defaultdict(list)
		for line in contents:

			elements = line.strip().split(", ")

			# dataset
			dataset = elements[0].strip()

			# (0) numUsers, 		(1) numItems, 		(2) numInteractions, 	(3) density
			# (4) userMin, 			(5) userMax, 		(6) userAvg, 			(7) itemMin, 		(8) itemMax, 		(9) itemAvg
			# (10) spaceSizeLog, 	(11) shapeLog, 		(12) densityLog, 		(13) userGini, 		(14) itemGini

			stats = [x.strip() for x in elements[1:]]
			stats = [int(float(x)) if int(float(x)) == float(x) else float(x) for x in stats]

			datasetStats[dataset] = stats

	return datasetStats


# Generate nicely formatted tables
def generateTables(datasetStats):

	# Basic & Detailed Statistics
	outFp = "../Datasets/characteristics_table_basic_detailed.txt"

	headerStr1 = ("|| {:<40s}" + " || " + " " * 48 + " || {:^33s}" * 2 + " ||\n").format(
		"", "Interactions per User", "Interactions per Item" )
	headerStr2 = ("|| {:<40s}" + " || {:^9s}" + " | {:^9s}" + " | {:^12s}" + " | {:^9s}" +
					" || {:^9s}" + " | {:^9s}" + " | {:^9s}" +
					" || {:^9s}" + " | {:^9s}" + " | {:^9s}" + " ||\n")
	headerStr2 = headerStr2.format( "Dataset", "Users", "Items", "Interactions", "Density",
		"Minimum", "Maximum", "Average", "Minimum", "Maximum", "Average")

	linebreak = "||" + "-" * 42 + "||" + "-" * 50 + "||" + "-" * 35 + "||" + "-" * 35 + "||\n"

	statsFmtStr = ("|| {:<40s}" + " || {:>9,d}" + " | {:>9,d}" + " | {:>12,d}" + " | {:>9.6f}" +
					" || {:>9,d}" + " | {:>9,d}" + " | {:>9.3f}" +
					" || {:>9,d}" + " | {:>9,d}" + " | {:>9.3f}" + " ||\n")

	with codecs.open(outFp, "w", encoding = "utf-8", errors = "ignore") as outFile:

		outFile.write("\n" + linebreak)
		outFile.write(headerStr1)
		outFile.write(headerStr2)
		outFile.write(linebreak)

		for idx, (dataset, stats) in enumerate(sorted(datasetStats.items())):

			datasetName = dataset.strip().replace("(Taste Profile Subset)", "(TPS)")
			outFile.write(statsFmtStr.format(
				*(["{:<4s} {:<s}".format( "({:d})".format( idx + 1 ), datasetName )] + stats[:10]) ))

		outFile.write(linebreak)


	# Basic & Advanced Statistics
	outFp = "../Datasets/characteristics_table_basic_advanced.txt"

	headerStr1 = ("|| {:<40s}" + " || {:^9s}" + " | {:^9s}" + " | {:^12s}" + " | {:^9s}" +
					" || {:^12s}" + " | {:^9s}" + " | {:^10s}" + " | {:^9s}" + " | {:^9s}" + " ||\n")
	headerStr1 = headerStr1.format( "Dataset", "Users", "Items", "Interactions", "Density",
		"spaceSizeLog", "shapeLog", "densityLog", "userGini", "itemGini")

	linebreak = "||" + "-" * 42 + "||" + "-" * 50 + "||" + "-" * 63 + "||\n"

	statsFmtStr = ("|| {:<40s}" + " || {:>9,d}" + " | {:>9,d}" + " | {:>12,d}" + " | {:>9.6f}" +
					" || {:>12.3f}" + " | {:>9.3f}" + " | {:>10.3f}" + " | {:>9.3f}" + " | {:>9.3f}" + " ||\n")

	with codecs.open(outFp, "w", encoding = "utf-8", errors = "ignore") as outFile:

		outFile.write("\n" + linebreak)
		outFile.write(headerStr1)
		outFile.write(linebreak)

		for idx, (dataset, stats) in enumerate(sorted(datasetStats.items())):

			datasetName = dataset.strip().replace("(Taste Profile Subset)", "(TPS)")
			outFile.write(statsFmtStr.format(
				*(["{:<4s} {:<s}".format( "({:d})".format( idx + 1 ), datasetName )] + stats[:4] + stats[10:]) ))

		outFile.write(linebreak)


def getDatasetsPapers(fp):

	contents = None
	with codecs.open(fp, "r", encoding = "utf-8", errors = "ignore") as inFile:
		contents = inFile.readlines()

	datasets_papers = defaultdict(list)
	for idx in range(0, len(contents), 3):

		dataset = contents[idx].strip().replace("https://", "").replace("www.", "")
		dataset = dataset.replace("Goodbooks", "Goodbooks-10k")
		lst_papers = sorted([int(x) for x in contents[idx + 1].strip().split(", ")])

		# These datasets are not publicly available
		if (dataset in ["Jester", "RecSys Challenge 2017 (i.e., XING)", "IMDb", "EachMovie"]):
			continue

		datasets_papers[dataset] = lst_papers

	return datasets_papers


