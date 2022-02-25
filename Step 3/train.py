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



if (__name__ == '__main__'):

	args = parse_args()

	# Initial Setup
	np.random.seed(args.random_seed)

	# Timer & Logging
	timer = Timer()
	timer.startTimer()

	uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

	# Dataset
	dataset = args.dataset.strip()
	args.dataDir = "../Datasets/Preprocessed/{}".format( dataset )

	# Logger
	uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
	logDir = "./logs/{}/{}/".format( dataset, args.model.strip() )
	logPath = "{}{}-{}".format(logDir, uuid, "logs.txt")
	logger = Logger(logDir, logPath, args)


	allUniqueUserIDs = np.genfromtxt(os.path.join(args.dataDir, "users.txt"), dtype = "str")
	numUsers = len(allUniqueUserIDs)

	allUniqueItemIDs = np.genfromtxt(os.path.join(args.dataDir, "items.txt"), dtype = "str")
	numItems = len(allUniqueItemIDs)

	logger.log("\nNumber of Users: {:,d}".format( numUsers ))
	logger.log("Number of Items: {:,d}".format( numItems ))


	# Load Training Data
	trainFp = os.path.join(args.dataDir, "train.csv")
	logger.log("\nLoading TRAINING data from \"{}\"..".format( trainFp ))

	trainData = loadTrainData(trainFp, numItems, confidence = (args.WMF_c_ui if args.model == "WMF" else 0))
	logger.log("Number of Training Samples: {:,d}".format( trainData.nnz ))
	logger.log("trainData's shape: {}".format( trainData.shape ))
	trainData = checkMatrix(trainData, format = "csr")

	logger.log("Training split loaded!")


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
			factors = args.WMF_factors,
			regularization = args.WMF_reg,
			iterations = args.WMF_iterations,
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
			topK = args.KNNCF_topK,
			shrink = args.KNNCF_shrink,
			similarity = args.KNNCF_similarity.strip().lower(),
			normalize = True if (args.KNNCF_normalize) else False,
			feature_weighting = args.KNNCF_feat_weight.strip()
		)

	elif (args.model == "RP3beta"):

		# Default: topK = 100, alpha = 1.0, beta = 0.6, normalize_similarity = False
		mdl.fit(
			topK = args.graph_topK,
			alpha = args.graph_alpha,
			beta = args.graph_beta,
			min_rating = 0,
			implicit = False,
			normalize_similarity = True if (args.graph_norm) else False
		)

	elif (args.model == "WMF"):

		mdl.fit(trainData)


	logger.log("'{}' fitted! {}".format( args.model, timer.getElapsedTimeStr("training", conv2HrsMins = True) ))


	# Load Validation & Testing Data
	validationFp, testFp = os.path.join(args.dataDir, "validation.csv"), os.path.join(args.dataDir, "test.csv")
	validationData, testData = loadValidationTestData(validationFp, testFp, numItems)

	logger.log("\nLoaded VALIDATION data from \"{}\"..".format( validationFp ))
	logger.log("Loaded TESTING data from \"{}\"..".format( testFp ))


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

	exit()


