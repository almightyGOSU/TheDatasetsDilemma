import os
import sys



TEXT_SEP = "-" * 140


# Creating a folder
def mkdir_p(path):

	if (path == ""):
		return
	try:
		os.makedirs(path)
	except:
		pass


# Saving all the arguments
def print_args(args, path = None):

	if (path):
		output_file = open(path, "w")

	args.command = " ".join(sys.argv)

	bHyperOpt = True if ("hyperOpt_train.py" in args.command) else False
	bLazyMain = True if ("MultiVAE_train.py" not in args.command) else False

	args.command = args.command.replace("clustering.py ", "")
	args.command = args.command.replace("hyperOpt_train.py ", "")
	args.command = args.command.replace("MultiVAE_train.py ", "")
	args.command = args.command.replace("train.py ", "")

	args.command = args.command.strip()

	items = vars(args)
	if (path):
		output_file.write("{}\n".format(TEXT_SEP))

	for key in sorted(items.keys(), key = lambda s: s.lower()):

		# Remove unwanted arguments...

		# Remove all hyperparameters when using Bayesian Hyperparameter Optimization
		if (bHyperOpt and any([key.startswith(x) for x in ["KNNCF", "graph", "WMF", "SLIM"]])):
			continue

		# Remove KNNCF-specific arguments if the model isn't UserKNNCF or ItemKNNCF
		if (bLazyMain and args.model not in ["UserKNNCF", "ItemKNNCF"] and key.startswith("KNNCF")):
			continue

		# Remove graph-specific arguments if the model isn't P3alpha or RP3beta
		if (bLazyMain and args.model not in ["P3alpha", "RP3beta"] and key.startswith("graph")):
			continue

		# Remove WMF-specific arguments if the model isn't WMF
		if (bLazyMain and args.model != "WMF" and key.startswith("WMF")):
			continue

		if (path):
			output_file.write("  " + key + ": " + str(items[key]) + "\n")

	if (path):

		output_file.write("{}\n".format(TEXT_SEP))
		output_file.close()

	print("\nCommand: {}".format( args.command ))
	del args.command


# Helper class for logging everything to "logs.txt"
class Logger():

	def __init__(self, out_dir, log_path = None, args = None):

		self.out_dir = out_dir

		if (log_path is None):
			self.log_path = "{}{}".format( out_dir, "logs.txt" )
		else:
			self.log_path = log_path

		mkdir_p(self.out_dir)
		with open(self.log_path, 'w+') as f:
			f.write("")

		if (args):
			print_args(args, path = self.log_path)


	def log(self, txt, log_path = None, print_txt = True):

		if (log_path is None):
			with open(self.log_path, 'a+') as f:
				f.write(txt + "\n")
		else:
			with open(log_path, 'a+') as f:
				f.write(txt + "\n")

		if (print_txt):
			print(txt)


	def logQ(self, txt, log_path = None):

		self.log(txt, log_path = log_path, print_txt = False)


	def emptyFile(self, log_path = None):
		with open(log_path, 'w+') as f:
			f.write("")


