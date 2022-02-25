from util import *

# This is for UserKNNCF, ItemKNNCF, and RP3beta
from model.Base.utilities import *

# These are for UserKNNCF, ItemKNNCF, and RP3beta
from model.UserKNNCF import UserKNNCFRecommender
from model.ItemKNNCF import ItemKNNCFRecommender
from model.RP3beta import RP3betaRecommender

from model.Parser import parse_args

# This is for Weighted MF (https://github.com/benfred/implicit)
import implicit

from model.Logger import Logger, TEXT_SEP
from model.Timer import Timer

from datetime import datetime


import traceback

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from functools import partial



def train(args, allUniqueUserIDs, allUniqueItemIDs, trainData, validationData, testData, hyperparamNames, hyperparamValues):

	try:

		hyperparameters = dict(zip(hyperparamNames, hyperparamValues))

		# Initial Setup
		np.random.seed(args.random_seed)

		# Timer & Logging
		timer = Timer()
		timer.startTimer()

		uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

		# Dataset
		dataset = args.dataset.strip()

		# Logger
		uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
		logDir = "./logs/{}/{}/".format( dataset, args.model )
		logPath = "{}{}-{}".format(logDir, uuid, "logs.txt")
		logger = Logger(logDir, logPath, args)


		# Store the current set of hyperparameters
		logger.log("\n>>> Hyperparameters <<<\n")
		for name, value in hyperparameters.items():

			formattedValue = value
			if (isinstance(formattedValue, int)):
				formattedValue = "{:,d}".format( formattedValue )
			elif (isinstance(formattedValue, float)):
				formattedValue = "{:.5f}".format( formattedValue ).rstrip("0").rstrip(".")

			logger.log("{:<30s} {}".format( "{}:".format( name ), formattedValue ))

		logger.log("\n{}".format( TEXT_SEP ), print_txt = False)


		# For WMF, multiply the training data with the confidence value
		if (args.model == "WMF"):
			trainData = trainData * hyperparameters["confidence"]


		# Statistics
		numUsers = len(allUniqueUserIDs)
		numItems = len(allUniqueItemIDs)

		logger.log("\nNumber of Users: {:,d}".format( numUsers ))
		logger.log("Number of Items: {:,d}".format( numItems ))

		logger.log("\nNumber of Training Samples: {:,d}".format( trainData.nnz ))
		logger.log("trainData's shape: {}".format( trainData.shape ))

		logger.log("\nNumber of Validation Samples: {:,d}".format( validationData.nnz ))
		logger.log("validationData's shape: {}".format( validationData.shape ))

		logger.log("\nNumber of Testing Samples: {:,d}".format( testData.nnz ))
		logger.log("testData's shape: {}".format( testData.shape ))


		# Start 'training'..
		logger.log("\n{}".format( TEXT_SEP ), print_txt = False)
		timer.startTimer("training")

		# Create Model
		print("")
		if (args.model == "UserKNNCF"):
			mdl = UserKNNCFRecommender(trainData)

		elif (args.model == "ItemKNNCF"):
			mdl = ItemKNNCFRecommender(trainData)

		elif (args.model == "RP3beta"):
			mdl = RP3betaRecommender(trainData)

		elif (args.model == "WMF"):
			mdl = implicit.als.AlternatingLeastSquares(
				factors = hyperparameters["factors"],
				regularization = hyperparameters["reg"],
				iterations = hyperparameters["iterations"],
				random_state = args.random_seed
			)

		else:
			print("Invalid Model! Please check your arguments..")
			exit()

		logger.log("\n\n'{}' created! {}".format( args.model, timer.getElapsedTimeStr("training", conv2HrsMins = True) ))


		# Fit Model
		logger.log("\nFitting '{}'..".format( args.model ))

		if (args.model in ["UserKNNCF", "ItemKNNCF"]):

			# Default: topK = 5, shrink = 0, similarity = "cosine", normalize = False, feature_weighting = "none"
			mdl.fit(
				topK = hyperparameters["topK"],
				shrink = hyperparameters["shrink"],
				similarity = hyperparameters["similarity"],
				normalize = hyperparameters["normalize"],
				feature_weighting = hyperparameters["feature_weighting"]
			)

		elif (args.model == "RP3beta"):

			# Default: topK = 100, alpha = 1.0, beta = 0.6, normalize_similarity = False
			mdl.fit(
				topK = hyperparameters["topK"],
				alpha = hyperparameters["alpha"],
				beta = hyperparameters["beta"],
				min_rating = 0,
				implicit = False,
				normalize_similarity = hyperparameters["normalize_similarity"]
			)

		elif (args.model == "WMF"):

			mdl.fit(trainData)

		logger.log("'{}' fitted! {}".format( args.model, timer.getElapsedTimeStr("training", conv2HrsMins = True) ))


		# Same values for validation & testing (under the Leave-One-Out evaluation strategy)
		numValidationUsers, numTestUsers = validationData.shape[0], testData.shape[0]
		validationBatchSize, testBatchSize = 1024, 1024


		# WMF is kinda dumb
		# The method fits on a 'items x users' matrix, but recommends using a 'users x items' matrix.. OTL
		if (args.model == "WMF"):
			trainData = trainData.T.tocsr()


		# Derive the scores for all users & items
		# For each user, the scores for previously consumed items are set to -inf
		maxN = max(VALIDATION_CUTOFF, max(rDictKV.keys()), max(nDictKV.keys()))
		userItemScores = getScores(args, mdl, trainData, numValidationUsers, validationBatchSize, maxN = maxN)
		logger.log("\nObtained all user-item scores!\t{}".format( timer.getElapsedTimeStr("training", conv2HrsMins = True) ))


		# Validation
		lstValidationNDCG = []
		batchIdx = 0
		for startIdx in tqdm(range(0, numValidationUsers, validationBatchSize), desc = "Validating"):

			endIdx = min(startIdx + validationBatchSize, numValidationUsers)
			lstValidationNDCG.append(validateBatchUsers(userItemScores[batchIdx], validationData[startIdx:endIdx]))

			batchIdx += 1

		validationNDCG = np.mean(np.concatenate(lstValidationNDCG))

		logger.log("\nValidation nDCG@10: {:.5f}\t{}".format( validationNDCG, timer.getElapsedTimeStr("training", conv2HrsMins = True) ))
		logger.log("\n\n<Best> Validation nDCG@10: {:.5f} (Epoch {})\n".format( validationNDCG, 1 ))


		# Testing
		lstTestOutputs = []
		batchIdx = 0
		for startIdx in tqdm(range(0, numTestUsers, testBatchSize), desc = "Testing"):

			endIdx = min(startIdx + testBatchSize, numTestUsers)
			lstTestOutputs.append(testBatchUsers(userItemScores[batchIdx], testData[startIdx:endIdx]))

			batchIdx += 1

		# lstTestOutputs is a list of 2 dictionaries, i.e. a list of (nDictResults, rDictResults)
		nListResultsFinal = defaultdict(list)
		rListResultsFinal = defaultdict(list)

		for nKey in nDictKV.keys():
			nListResultsFinal[nKey] = np.concatenate([x[0][nKey] for x in lstTestOutputs], axis = None)

		for rKey in rDictKV.keys():
			rListResultsFinal[rKey] = np.concatenate([x[1][rKey] for x in lstTestOutputs], axis = None)

		N = np.sqrt(len(nListResultsFinal[5]))

		logger.log("\n")
		for nKey in sorted(nDictKV.keys()):
			logger.log("{:15s} = {:.5f} ({:.5f})".format(
				"Test {}".format( nDictKV[nKey] ),
				np.mean(nListResultsFinal[nKey]),
				np.std(nListResultsFinal[nKey]) / N ))

		logger.log("")
		for rKey in sorted(rDictKV.keys()):
			logger.log("{:15s} = {:.5f} ({:.5f})".format(
				"Test {}".format( rDictKV[rKey] ),
				np.mean(rListResultsFinal[rKey]),
				np.std(rListResultsFinal[rKey]) / N ))


		logger.log("")
		logger.log("End of Program!\t{}\n".format( timer.getElapsedTimeStr("training", conv2HrsMins = True) ))

		# Note: Negation (To maximize nDCG, you need to minimize the negated nDCG)
		return -validationNDCG

	except (KeyboardInterrupt, SystemExit) as e:

		# If getting a interrupt, terminate without saving the exception
		raise e

	except:

		# Catch any error: Exception, Tensorflow errors, etc...
		traceback_string = traceback.format_exc()
		logger.log("\n*** *** [ERROR] *** ***\n{}\n".format( traceback_string ))
		traceback.print_exc()

		# Set to the largest negative value for a float
		return -np.finfo(np.float32).max



if (__name__ == '__main__'):

	args = parse_args()

	# Dataset
	dataset = args.dataset.strip()
	args.dataDir = "../Datasets/Preprocessed/{}".format( dataset )

	allUniqueUserIDs = np.genfromtxt(os.path.join(args.dataDir, "users.txt"), dtype = "str")
	allUniqueItemIDs = np.genfromtxt(os.path.join(args.dataDir, "items.txt"), dtype = "str")

	numUsers = len(allUniqueUserIDs)
	numItems = len(allUniqueItemIDs)

	# Load Training Data
	trainFp = os.path.join(args.dataDir, "train.csv")
	trainData = loadTrainData(trainFp, numItems, confidence = 1 if args.model == "WMF" else 0) # Non-zero confidence level transposes trainData
	trainData = checkMatrix(trainData, format = "csr")

	# Load Validation & Testing Data
	validationFp, testFp = os.path.join(args.dataDir, "validation.csv"), os.path.join(args.dataDir, "test.csv")
	validationData, testData = loadValidationTestData(validationFp, testFp, numItems)


	# Hyperparameters Names & Search Space
	if (args.model in ["UserKNNCF", "ItemKNNCF"]):

		hyperparamNames = ["topK", "shrink", "similarity", "normalize", "feature_weighting"]
		hyperparamSpace = [Integer(5, 1000, name = "topK"),
							Integer(5, 1000, name = "shrink"),
							Categorical(["cosine"], name = "similarity"),
							Categorical([True, False], name = "normalize"),
							Categorical(["none"], name = "feature_weighting")]

	elif (args.model == "RP3beta"):

		hyperparamNames = ["topK", "alpha", "beta", "normalize_similarity"]
		hyperparamSpace = [Integer(5, 1000, name = "topK"),
							Real(low = 0, high = 2, prior = "uniform", name = "alpha"),
							Real(low = 0, high = 2, prior = "uniform", name = "beta"),
							Categorical([True, False], name = "normalize_similarity")]

	elif (args.model == "WMF"):

		hyperparamNames = ["confidence", "factors", "reg", "iterations"]
		hyperparamSpace = [Integer(2, 100, name = "confidence"),
							Integer(100, 200, name = "factors"),
							Categorical([0.01], name = "reg"),
							Categorical([15], name = "iterations")]

	else:
		print("Invalid model!")
		exit()


	# Run Bayesian Hyperparamters Optimization; A total of 35 times, with the first 5 times being random search
	train = partial(train, args, allUniqueUserIDs, allUniqueItemIDs, trainData.copy(), validationData, testData, hyperparamNames)
	res_gp = gp_minimize(train, hyperparamSpace, n_calls = 35, n_initial_points = 5, random_state = args.random_seed, noise = 1e-10)


	print("")
	exit()


