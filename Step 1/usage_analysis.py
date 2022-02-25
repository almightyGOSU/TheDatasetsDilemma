import codecs
import numpy as np
import pandas as pd

from collections import defaultdict, OrderedDict, Counter

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax



def getDatasetsPapers(fp = "./datasets_to_papers.txt"):

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

	print("[getDatasetsPapers] # of datasets: {:,d}".format( len(datasets_papers )))

	return datasets_papers


def getPapersDatasets(fp = "./datasets_to_papers.txt"):

	contents = None
	with codecs.open(fp, "r", encoding = "utf-8", errors = "ignore") as inFile:
		contents = inFile.readlines()

	papers_datasets = defaultdict(list)
	for idx in range(0, len(contents), 3):

		dataset = contents[idx].strip().replace("https://", "").replace("www.", "")
		dataset = dataset.replace("Goodbooks", "Goodbooks-10k")
		lst_papers = sorted([int(x) for x in contents[idx + 1].strip().split(", ")])

		# These datasets are not publicly available
		if (dataset in ["Jester", "RecSys Challenge 2017 (i.e., XING)", "IMDb", "EachMovie"]):
			continue

		for paper in lst_papers:
			papers_datasets[paper].append(dataset)

	return papers_datasets


def getPapersToCitations(fp = "./papers_to_citations.txt"):

	contents = None
	with codecs.open(fp, "r", encoding = "utf-8", errors = "ignore") as inFile:
		contents = inFile.readlines()

	papers_citations = defaultdict(str)
	for idx, line in enumerate(contents):

		papers_citations[idx + 1] = line.strip()

	print("[getPapersToCitations] # of papers: {:,d}".format( len(papers_citations )))

	return papers_citations


def getPapersToCitedPapers(fp = "./papers_to_cited_papers.txt"):

	contents = None
	with codecs.open(fp, "r", encoding = "utf-8", errors = "ignore") as inFile:
		contents = inFile.readlines()

	papers_citedPapers = defaultdict(str)
	for idx, line in enumerate(contents):

		papers_citedPapers[idx + 1] = line.strip()

	print("[getPapersToCitedPapers] # of papers: {:,d}".format( len(papers_citedPapers )))

	return papers_citedPapers



if (__name__ == '__main__'):

	# Mapping of datasets to the set of papers which utilised each of these datasets
	datasets_papers = getDatasetsPapers()

	# Sort the datasets based on their usage frequencies (tie-breaker: alphabetical order of dataset name)
	datasets_papers = OrderedDict(sorted(datasets_papers.items(), key = lambda x: (-len(x[1]), x[0]), reverse = False))

	# Mapping of dataset to its zero-indexed ID
	datasets_datasetIDs = {dataset: (len(datasets_papers) - idx) for idx, dataset in enumerate(datasets_papers.keys())}

	# For mapping paper from ID to the citation command
	papers_citations = getPapersToCitations()

	# For mapping paper from ID to the cited paper
	papers_citedPapers = getPapersToCitedPapers()


	all_papers = []
	for lst_papers in datasets_papers.values():
		all_papers.extend(lst_papers)

	paperCounter = Counter(all_papers)
	paperIDs_xticks = {paperID: (idx + 1) for idx, (paperID, _) in enumerate(
		sorted(paperCounter.items(), key = lambda x: (-x[1], x[0]), reverse = False))}

	print("")
	lstCounts = []
	countToPapers = defaultdict(list)
	for paper, count in paperCounter.items():

		lstCounts.append(count)
		countToPapers[count].append(paper)

	NUM_DATASETS = len(datasets_papers)
	NUM_PAPERS = len(set().union(*datasets_papers.values()))
	TOTAL_FREQUENCY = sum(len(lst_papers) for lst_papers in datasets_papers.items())


	# ====================================================================================================
	# Scatter Plot (Usage Analysis)
	# x-axis: Papers
	# y-axis: Datasets
	plotTitle = "Usage Patterns of Recommendation Datasets in Recent Papers"

	# Default fig size is [6.4, 4.8] (in inches, not pixels)
	mainfig = plt.figure(figsize = (24.0, 15.0), dpi = 300)
	ax = plt.gca()

	# Coloring scheme based on usage frequency
	# >= 5 papers 				---> Green
	# > 1 paper and < 5 papers 	---> Blue
	# 1 paper 					---> Red

	maxPaperID = 0
	for dataset, lst_papers in datasets_papers.items():

		datasetID = datasets_datasetIDs[dataset]
		numPapers = len(lst_papers)

		for paperID in lst_papers:

			plt.plot(paperIDs_xticks[paperID], datasetID,
				"go" if numPapers >= 5 else ("ro" if numPapers == 1 else "bo"),
				markersize = 10 if numPapers >= 5 else (8 if numPapers == 1 else 9))

			maxPaperID = max(maxPaperID, paperID)

	# y-ticks and labels (Datasets)
	ax.set_yticks(list(datasets_datasetIDs.values()))
	ax.set_yticklabels(datasets_datasetIDs.keys(), fontsize = 16)

	# Customize colors for y-labels
	for dataset, ylabel in zip(datasets_datasetIDs.keys(), ax.yaxis.get_ticklabels()):

		numPapers = len(datasets_papers[dataset])
		ylabel.set_color("green" if numPapers >= 5 else ("red" if numPapers == 1 else "blue"))
		ylabel.set_weight("extra bold" if numPapers >= 5 else ("normal" if numPapers == 1 else "bold"))
		ylabel.set_style("italic" if numPapers >= 5 else "normal")


	lstXTickLabels = list(paperIDs_xticks.keys())
	lstXTickLabels = [papers_citedPapers[x] for x in lstXTickLabels]

	# x-ticks and labels (Papers)
	ax.set_xticks(range(1, maxPaperID + 1))
	ax.set_xticklabels(lstXTickLabels, fontsize = 16)

	# x-limits and y-limits
	plt.xlim(0, maxPaperID + 1)
	plt.ylim(0, len(datasets_datasetIDs) + 1)

	# Rotate x-labels
	plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")

	ax.set_xlabel("Recent Papers", fontsize = 24, fontweight = "bold", labelpad = 10)


	# Grid
	for xmajor in ax.xaxis.get_majorticklocs():
		ax.axvline(x = xmajor, ls = '--', alpha = 0.6, linewidth = 0.6, color = "grey", zorder = -1)
	for ymajor in ax.yaxis.get_majorticklocs():
		ax.axhline(y = ymajor, ls = '--', alpha = 0.6, linewidth = 0.6, color = "grey", zorder = -1)


	# Dataset Frequencies
	lst5OrMore, lst2to4, lst1 = [],[],[]
	for dataset, lst_papers in datasets_papers.items():

		datasetID = datasets_datasetIDs[dataset]
		numPapers = len(lst_papers)

		if (numPapers >= 5):
			lst5OrMore.append(datasetID)
		elif(numPapers >= 2):
			lst2to4.append(datasetID)
		else:
			lst1.append(datasetID)

		plt.text(ax.get_xlim()[1] + 0.5, datasetID, "{:<2d}".format( numPapers ),
			fontsize = 16,
			fontweight = "extra bold" if numPapers >= 5 else ("normal" if numPapers == 1 else "bold"),
			fontstyle = "italic" if numPapers >= 5 else "normal",
			color = "green" if numPapers >= 5 else ("red" if numPapers == 1 else "blue"))


	# Save as .png file
	plt.gcf().subplots_adjust(top = 0.98, left = 0.22, right = 0.95, bottom = 0.08)
	fp_image_file = "Datasets Usage (Scatter).png"
	mainfig.savefig(fp_image_file)
	plt.close(mainfig)


	print("{:>2d}/{:d} datasets ({:.2f}%) are used in 5 or more papers..".format(
		len(lst5OrMore), NUM_DATASETS, len(lst5OrMore) / NUM_DATASETS * 100 ))
	print("{:>2d}/{:d} datasets ({:.2f}%) are used in 2 to 4 papers..".format(
		len(lst2to4), NUM_DATASETS, len(lst2to4) / NUM_DATASETS * 100 ))
	print("{:>2d}/{:d} datasets ({:.2f}%) are used in 1 paper..".format(
		len(lst1), NUM_DATASETS, len(lst1) / NUM_DATASETS * 100 ))



	# ====================================================================================================
	# Find the most frequent dataset combinations
	papers_datasets = getPapersDatasets()

	lstPapersDatasets = []
	for paper, datasets in sorted(papers_datasets.items()):
		lstPapersDatasets.append(datasets)

	te = TransactionEncoder()
	te_ary = te.fit(lstPapersDatasets).transform(lstPapersDatasets)
	df = pd.DataFrame(te_ary, columns = te.columns_)

	frequent_itemsets = apriori(df, min_support = (2.0 / 48.0), use_colnames = True)
	frequent_itemsets["Number of Items"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
	frequent_itemsets["Number of Papers"] = frequent_itemsets["support"].apply(lambda x: int(x * len(papers_datasets)))
	frequent_itemsets = frequent_itemsets[(frequent_itemsets["Number of Items"] >= 2)]

	frequent_itemsets.sort_values(["Number of Items", "Number of Papers"], ascending = [False, False], inplace = True)
	frequent_itemsets.reset_index(drop = True, inplace = True)


	with codecs.open("frequent_dataset_combinations.txt", "w", encoding = "utf-8", errors = "ignore") as outFile:

		outFile.write("|| {:^60s} | {:^20s} | {:^40s} | {:^20s} ||\n".format(
			"Datasets", "Number of Datasets", "Papers", "Number of Papers" ))

		linebreak = "||" + "-" * 62 + "|" + "-" * 22 + "|" + "-" * 42 + "|" + "-" * 22 + "||\n"
		outFile.write(linebreak)

		fmtStr = ("|| {:<60s}" + " | {:>20d}" + " | {:<40s}" + " | {:>20d}" + " ||\n")

		currNumDatasets = 0
		encounteredItemsets = defaultdict(list)

		for row in frequent_itemsets.itertuples(index = False, name = "Pandas"):

			numPapers = int(row.support * NUM_PAPERS)

			lstDatasets = sorted(list(row.itemsets))
			numDatasets = len(lstDatasets)

			lstPapersForCurrDatasets = sorted(list(set.intersection(
				*[set(papers) for dataset, papers in datasets_papers.items() if dataset in lstDatasets] )))

			bDup = False
			for encounteredItemset, lstPapersForEncounteredItemset in encounteredItemsets.items():
				if (set(lstDatasets) <= set(encounteredItemset) and (lstPapersForCurrDatasets == lstPapersForEncounteredItemset)):
					bDup = True
					break

			if (bDup): continue

			if (currNumDatasets != 0 and currNumDatasets != numDatasets):
				currNumDatasets = numDatasets
				outFile.write(linebreak)

			if (currNumDatasets == 0):
				currNumDatasets = numDatasets

			outLst = sorted([int(papers_citedPapers[x].strip().replace("[", "").replace("]", "")) for x in lstPapersForCurrDatasets])
			outFile.write(fmtStr.format( ", ".join(lstDatasets), numDatasets, ", ".join([str(x) for x in outLst]), numPapers ))

			encounteredItemsets[tuple(lstDatasets)] = lstPapersForCurrDatasets

		outFile.write(linebreak)
		outFile.write("\n\n\n")


		# LaTeX Table
		outFile.write("{:s} & {:s}\\\\\n".format( "Datasets", "Papers" ))

		linebreak = "\\midrule\n"
		outFile.write(linebreak)

		fmtStr = ("{:s} & {:s}\\\\\n")

		currNumDatasets = 0
		encounteredItemsets = defaultdict(list)

		for row in frequent_itemsets.itertuples(index = False, name = "Pandas"):

			numPapers = int(row.support * NUM_PAPERS)

			lstDatasets = sorted(list(row.itemsets))
			numDatasets = len(lstDatasets)

			lstPapersForCurrDatasets = sorted(list(set.intersection(
				*[set(papers) for dataset, papers in datasets_papers.items() if dataset in lstDatasets] )))

			bDup = False
			for encounteredItemset, lstPapersForEncounteredItemset in encounteredItemsets.items():
				if (set(lstDatasets) <= set(encounteredItemset) and (lstPapersForCurrDatasets == lstPapersForEncounteredItemset)):
					bDup = True
					break

			if (bDup): continue

			if (currNumDatasets != 0 and currNumDatasets != numDatasets):
				currNumDatasets = numDatasets
				outFile.write(linebreak)

			if (currNumDatasets == 0):
				currNumDatasets = numDatasets

			outFile.write(fmtStr.format( ", ".join(lstDatasets),
				"{}{}{}".format( "\\cite{", ",".join([str(papers_citations[x]) for x in lstPapersForCurrDatasets]), "}") ))

			encounteredItemsets[tuple(lstDatasets)] = lstPapersForCurrDatasets

		outFile.write(linebreak)



	print("")
	exit()


