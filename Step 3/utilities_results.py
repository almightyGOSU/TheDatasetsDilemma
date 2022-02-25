import re
import os

from collections import defaultdict



# Creating a folder
def mkdir_p(path):

	if path == "":
		return
	try:
		os.makedirs(path)
	except:
		pass


# Check for NaN
def checkForNaN(inFile):

	lastLine = inFile.readlines()[-1]
	if ("nan" in lastLine):
		return True
	return False


# Returns the statistics obtained at the best epoch
def processDoc(inFile, args):

	hyperParamNames, hyperParamValues = None, None

	bestEpoch = 0
	ln_bestEpoch = 0

	doc = inFile.readlines()
	for line_num, line in enumerate(doc):

		text = line.strip()

		# Hyperparameters
		if (text == ">>> Hyperparameters <<<"):

			hyperParamNames, hyperParamValues = [], []
			for line2 in doc[(line_num + 2):]:

				hyperParamsText = line2.strip()

				if (not hyperParamsText):
					break

				name, value = hyperParamsText.split(":")
				hyperParamNames.append(name.strip())
				hyperParamValues.append(value.strip())

			hyperParamNames = ", ".join(hyperParamNames)
			hyperParamValues = ", ".join(hyperParamValues)

		# Best Validation nDCG@10
		if (text.startswith("<Best> Validation nDCG@10:")):

			contents = text.split(":")[1].strip()
			contents = contents.replace("(Epoch ", "").replace(")", "")
			contents = contents.split(" ")

			validationScore, bestEpoch = float(contents[0]), int(contents[1])
			ln_bestEpoch = line_num

			break

	'''
	<Best> Validation nDCG@10: 0.29137 (Epoch 1) 		ln_bestEpoch

	Test nDCG@5     = 0.28731 (0.00284) 				+ 4
	Test nDCG@10    = 0.27935 (0.00243) 				+ 5
	Test nDCG@15    = 0.28358 (0.00228)
	Test nDCG@20    = 0.29081 (0.00221)
	Test nDCG@25    = 0.29861 (0.00218)
	Test nDCG@50    = 0.33204 (0.00212)
	Test nDCG@75    = 0.35536 (0.00212)
	Test nDCG@100   = 0.37180 (0.00211) 				+ 11

	Test Recall@5   = 0.27735 (0.00274) 				+ 13
	Test Recall@10  = 0.28672 (0.00253) 				+ 14
	Test Recall@15  = 0.31186 (0.00257)
	Test Recall@20  = 0.33876 (0.00264)
	Test Recall@25  = 0.36456 (0.00272)
	Test Recall@50  = 0.46466 (0.00288)
	Test Recall@75  = 0.53237 (0.00289)
	Test Recall@100 = 0.58128 (0.00285) 				+ 20
	'''

	if (ln_bestEpoch != 0):

		lst_ndcg = [process_value(text) for text in doc[(ln_bestEpoch + 4):(ln_bestEpoch + 12)]]
		lst_recall = [process_value(text) for text in doc[(ln_bestEpoch + 13):(ln_bestEpoch + 21)]]

		# Check that we have nDCG for all cut-offs and Recall for all cut-offs
		if (len(lst_ndcg) < 8 or len(lst_recall) < 8):
			bestEpoch = None

	else:

		return hyperParamNames, hyperParamValues, None, None, None, None

	return hyperParamNames, hyperParamValues, validationScore, bestEpoch, lst_ndcg, lst_recall


def process_value(text):

	if (text.strip() == ""):
		return None

	text = text.split("=")[1].strip()
	text = text.replace("(", "").replace(")", "")

	contents = text.split(" ")
	mean, stdDev = float(contents[0]), float(contents[1])

	return (mean, stdDev)


# Sort the command based on the 'keys'
def arrange_command(command):

	parts = command.split()
	# print("len(parts): {}".format(len(parts)))

	lstOptions = []
	for i in range(int(len(parts) / 2)):

		opt = parts[i * 2]
		val = parts[i * 2 + 1]

		opt = opt.replace("--", "-")

		lstOptions.append( (opt, val) )
		# print("{} {}".format(opt, val))

	lstOptions = sorted(lstOptions, key = lambda option: option[0])
	sortedCmd = " ".join(["{} {}".format(opt[0], opt[1]) for opt in lstOptions])

	return sortedCmd.strip()


def split_for_search(command, prefix = None):

	parts = command.split()
	lstOptVal = {}

	for i in range( int(len(parts) / 2) ):

		opt = parts[i * 2]
		val = parts[i * 2 + 1]

		if prefix and opt.startswith(prefix):
			lstOptVal[opt] = val

	return sorted(lstOptVal.keys()), lstOptVal


def processCommand(text, args):

	# Remove "command:"
	command = text.split(":")[1].strip()

	# Clean-up
	command = command.replace("hyperOpt_train.py ", "")
	command = command.replace("MultiVAE_train.py ", "")
	command = command.replace("train.py ", "")

	command = command.replace("--dataset", "-d")
	command = command.replace("-dataset", "-d")
	command = command.replace("--model", "-m")

	command = command.replace("--batch_size", "-bs")
	command = command.replace("--epochs", "-e")

	command = re.sub(r"--gpu [0-7]", "", command)
	command = re.sub(r"-gpu [0-7]", "", command)

	# Remove dataset information from command
	command = command.replace("-d {}".format( args.dataset ), "")

	# Remove model information from command
	command = command.replace("-m {}".format( args.model ), "")

	# Final processing
	command = arrange_command(command.strip())

	return command


