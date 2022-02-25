from util import *

from model.Base.utilities import *
from model.MultiVAE import MultiVAE
from model.Timer import Timer
from model.Logger import Logger, TEXT_SEP

import argparse

from datetime import datetime

import tensorflow as tf
import tensorflow.keras as tfk



if __name__ == "__main__":

	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

	parser = argparse.ArgumentParser(description="")
	parser.add_argument("-dataset", 		type = str, 	default = "ml-20m", 	help = "Dataset, e.g. ml-20m, millionsong")
	parser.add_argument("-batch_size", 		type = int, 	default = 500)
	parser.add_argument("-n_epochs", 		type = int, 	default = 200)

	# Number of Hidden Layers, i.e. {0, 1, 2}
	parser.add_argument("-num_hidden", 		type = int, 	default = 1)

	# Hyperparameter \beta for KL annealing, i.e. [0.1, 1.0]
	parser.add_argument("-beta", 			type = float, 	default = 0.2)

	parser.add_argument("-early_stop", 		type = int, 	default = 20,
		help = "Early Stopping, if performance does not improve after X epochs (Default: 20, i.e. 20 successive validation steps)")

	parser.add_argument("-random_seed", 	type = int, 	default = 1337, 		help = "Random Seed (Default: 1337)")

	args = parser.parse_args()


	# Initial Setup
	np.random.seed(args.random_seed)
	tf.random.set_seed(args.random_seed)

	# Dataset
	dataset = args.dataset.strip()

	args.dataDir = 		"../Datasets/Preprocessed/{}".format( dataset )
	args.chkpt_dir = 	"./chkpt/{}/vaecf".format( dataset )

	if not os.path.isdir(args.chkpt_dir):
		os.makedirs(args.chkpt_dir)


	# Logger
	uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
	log_dir = "./logs/{}/MultiVAE/".format( dataset )
	log_path = "{}{}-{}".format(log_dir, uuid, "logs.txt")
	logger = Logger(log_dir, log_path, args)

	# Store the current set of hyperparameters
	logger.log("\n>>> Hyperparameters <<<\n")
	logger.log("{:<30s} {}".format( "{}:".format( "epochs" ), args.n_epochs ))
	logger.log("{:<30s} {}".format( "{}:".format( "hidden" ), args.num_hidden ))
	logger.log("{:<30s} {}".format( "{}:".format( "beta" ), args.beta ))
	logger.log("\n{}".format( TEXT_SEP ), print_txt = False)


	allUniqueUserIDs = np.genfromtxt(os.path.join(args.dataDir, "users.txt"), dtype = "str")
	allUniqueItemIDs = np.genfromtxt(os.path.join(args.dataDir, "items.txt"), dtype = "str")

	numUsers = len(allUniqueUserIDs)
	numItems = len(allUniqueItemIDs)

	# Load Training Data
	trainFp = os.path.join(args.dataDir, "train.csv")
	trainData = loadTrainData(trainFp, numItems)
	trainData = checkMatrix(trainData, format = "csr")

	# Load Validation & Testing Data
	validationFp, testFp = os.path.join(args.dataDir, "validation.csv"), os.path.join(args.dataDir, "test.csv")
	validationData, testData = loadValidationTestData(validationFp, testFp, numItems)


	N = trainData.shape[0]
	idx = np.arange(N)
	n_epochs = args.n_epochs
	batch_size = args.batch_size

	logger.log("\nTraining data loaded from '{}'..".format( trainFp ))
	logger.log("Number of Training Samples: {:,d}".format( trainData.nnz ))
	logger.log("trainData's shape: {}".format( trainData.shape ))

	N_vald = validationData.shape[0]
	batch_size_vald = 2000


	total_anneal_steps = 200000
	anneal_cap = args.beta
	dropout_prob = 0.5
	p_dims = [200] + ([600] * args.num_hidden) + [numItems]


	# Create Model
	vae = MultiVAE(p_dims)

	# Optimizer
	optimizer = tfk.optimizers.Adam()

	@tf.function
	def train_step(X, anneal_factor):
		with tf.GradientTape() as tape:
			logits, KLD = vae(X, training = True)
			logPr = tf.nn.log_softmax(logits, axis = 1)
			negLL = -tf.reduce_mean(tf.reduce_sum(X * logPr, axis = 1))
			loss = negLL + anneal_factor * KLD
		gradients = tape.gradient(loss, vae.trainable_variables)
		optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

	@tf.function
	def test_step(X):
		logits, _ = vae(X, training = False)
		return logits


	# Timer
	timer = Timer()

	best_epoch = 0
	best_ndcg = -np.inf
	update_count = 0

	logger.log("\nStart training...")
	timer.startTimer("training")

	for epoch in range(n_epochs):

		np.random.shuffle(idx)

		for st_idx in tqdm(range(0, N, batch_size), desc = "Training"):

			end_idx = min(st_idx + batch_size, N)

			X = trainData[idx[st_idx:end_idx]]

			if sparse.isspmatrix(X):
				X = X.toarray()

			X = X.astype(np.float32)

			anneal_factor = min(anneal_cap, update_count / total_anneal_steps)
			train_step(X, tf.constant(anneal_factor))
			update_count += 1

		logger.logQ("")
		logger.log("{:15s} {:24s}\t{}".format(
			"[Epoch {:d}/{:d}]".format( epoch + 1, n_epochs ),
			"Training Step Completed",
			timer.getElapsedTimeStr("training", conv2HrsMins = True) ))

		ndcg_list = []
		for st_idx in range(0, N_vald, batch_size_vald):

			end_idx = min(st_idx + batch_size_vald, N_vald)

			X = trainData[st_idx:end_idx]
			if sparse.isspmatrix(X):
				X = X.toarray()
			X = X.astype(np.float32)

			logits = test_step(X)
			logits = logits.numpy()
			logits[X.nonzero()] = -np.inf
			ndcg_list.append(ndcg_at_k(logits, validationData[st_idx:end_idx].toarray(), k = VALIDATION_CUTOFF))

		ndcg = np.mean(np.concatenate(ndcg_list))

		logger.log("{:15s} Validation nDCG@10: {:.5f}\t{}".format(
			"[Epoch {:d}/{:d}]".format( epoch + 1, n_epochs ),
			ndcg, timer.getElapsedTimeStr("training", conv2HrsMins = True) ))

		if (ndcg > best_ndcg):

			best_epoch = epoch
			best_ndcg = ndcg

			vae.save_weights(os.path.join(args.chkpt_dir, "model"), save_format = "tf")
			logger.log("{:15s} Validation nDCG@10: {:.5f}\t<Best> \\o/\\o/\\o/".format( "[Epoch {}]".format( epoch + 1 ), ndcg ))

		# [Optional] Early Stopping
		if (args.early_stop > 0 and epoch - best_epoch >= args.early_stop):

			logger.log("\n>>> MODEL performance, in terms of the best validation nDCG@10, has stopped improving!")
			logger.log(">>> Best validation nDCG@10 of {:.5f} was obtained after training for {:d} epochs!".format(
				best_ndcg, best_epoch + 1 ))
			logger.log(">>> Now, validation nDCG@10 of {:.5f}  is obtained after training for {:d} epochs!".format(
				ndcg, epoch + 1 ))
			logger.log(">>> Given that there is NO improvement after {:d} successive epochs, " \
				"we are prematurely stopping the model!!!".format( args.early_stop ))

			# EARLY STOPPING TRIGGERED
			break


	# Best Validation nDCG & Epoch
	logger.log("\n\n<Best> Validation nDCG@10: {:.5f} (Epoch {})\n".format( best_ndcg, best_epoch + 1 ))


	# Evaluation - Testing
	N_test = testData.shape[0]
	batch_size_test = 2000

	# Load back the 'best' model weights for testing
	best_model_path = os.path.join(args.chkpt_dir, "model")
	vae.load_weights(best_model_path)

	# Testing
	nList = defaultdict(list)
	rList = defaultdict(list)

	for st_idx in range(0, N_test, batch_size_test):

		end_idx = min(st_idx + batch_size_test, N_test)

		X = trainData[st_idx:end_idx]

		if sparse.isspmatrix(X):
			X = X.toarray()
		X = X.astype(np.float32)

		logits = test_step(X)
		logits = logits.numpy()
		logits[X.nonzero()] = -np.inf

		Z = testData[st_idx:end_idx].toarray()

		for nKey in nDictKV.keys():
			nList[nKey].append(ndcg_at_k(logits, Z, k = nKey))

		for rKey in rDictKV.keys():
			rList[rKey].append(recall_at_k(logits, Z, k = rKey))


	for nKey in nDictKV.keys():
		nList[nKey] = np.concatenate(nList[nKey])

	for rKey in rDictKV.keys():
		rList[rKey] = np.concatenate(rList[rKey])

	N = np.sqrt(len(nList[5]))

	logger.log("\n")
	for nKey in sorted(nDictKV.keys()):
		logger.log("{:15s} = {:.5f} ({:.5f})".format( "Test {}".format( nDictKV[nKey] ), np.mean(nList[nKey]), np.std(nList[nKey]) / N ))

	logger.log("")
	for rKey in sorted(rDictKV.keys()):
		logger.log("{:15s} = {:.5f} ({:.5f})".format( "Test {}".format( rDictKV[rKey] ), np.mean(rList[rKey]), np.std(rList[rKey]) / N ))


	# Log info
	logger.log("\nTesting Step Completed\t{}".format( timer.getElapsedTimeStr("training", conv2HrsMins = True) ))
	logger.log("\n\nModel w/ the best validation nDCG@10 of '{:.5f}' was loaded from '{}'..\n".format( best_ndcg, best_model_path ))


