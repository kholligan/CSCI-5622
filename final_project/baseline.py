import argparse
from collections import Counter, defaultdict

import random
import math
import scipy.stats as sp
import numpy as np
import gzip
import re

class Data:
	"""
	Class to hold the data set.
	"""

	def __init__(self, location, split_ratio, limit=None):

		# self.dataset = defaultdict(list)
		self.dataset = []
		self.split_ratio = split_ratio
		self.train_x = []
		self.train_y = []
		self.valid_x = []
		self.valid_y = []
		self.limit = limit

		with gzip.open(location, 'rb') as f:
			for line in f:
				x = line.decode('utf8').splitlines()
				for item in x:
					self.dataset.append(item.split(','))
		self.splitDataset()

	def splitDataset(self):
		if self.limit is None:
			train_size = int(len(self.dataset)*self.split_ratio)
			valid_size = int(len(self.dataset)*(1-self.split_ratio))
		else:
			train_size = self.limit
			valid_size = self.limit*(1-self.split_ratio)

		# print(len(self.dataset), self.split_ratio, train_size)
		copy = self.dataset
		print(len(copy))
		# Randomly pull items from the data set for the training set
		while len(self.train_x) < train_size:
			index = random.randrange(len(copy))
			item = copy.pop(index)
			self.train_x.append(item)
			# self.train_x.append(item[:-1])
			# self.train_y.append(item[-1])

		# Put the remaining items in the validation set
		while len(self.valid_x) < valid_size:
			index = random.randrange(len(copy))
			item = copy.pop()
			self.valid_x.append(item)
			# self.valid_x.append(item[:-1])
			# self.valid_y.append(item[-1])
		print(len(self.train_x))
		print(len(self.valid_x))

class NaiveBayes:
	"""
	Class to compute naive bayes probabilities
	"""

	def __init__(self, data):
		# 41 features
		# self.data_headers = ['duration', 'protocol_type', 'service', 'flag',
		# 	'src_bytes','dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
		# 	'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
		# 	'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
		# 	'num_access_files', 'num_outbound_cmds', 'is_hot_login', 
		# 	'is_guest_login', 'count', 'srv_count', 'serror_rate', 
		# 	'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
		# 	'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
		# 	'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
		# 	'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
		# 	'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
		# self.data_vector = dict.fromkeys(self.data_headers)
		self.dataset = data

	def create_class_vectors(self, dataset):
		'''
		Create a dictionary of lists for each class in the data set
		'''
		class_vector = {}
		for i in range(len(dataset)):
			feature = dataset[i]
			if(feature[-1] not in class_vector):
				class_vector[feature[-1]] = []
			class_vector[feature[-1]].append(feature)
		return class_vector

	def calculate_feature_values(self, dataset):
		'''
		Calculate the mean and std deviation for each feature in a class
		'''
		class_vals = []
		for attribute in zip(*dataset):
			if attribute[0].isdigit() is True:
				class_vals.append((np.mean(list(map(int, attribute))), np.std(list(map(int, attribute)))))
			else:
				class_vals.append((0,0))
		del class_vals[-1]
		return class_vals

	def summarize_features_by_class(self, dataset):
		'''
		Summarize the feature values by class
		'''
		class_vector = self.create_class_vectors(dataset)
		summaries = {}
		for class_label, features in class_vector.items():
			summaries[class_label] = self.calculate_feature_values(features)
		return summaries

	def calculate_normal_PDF(self, x, mean, stdev):
		'''
		Calculate the gaussian probability density function.
		Probability that a feature belongs to a class
		'''
		# Getting an error when I try to us scipy.stats.norm()
		x = int(x)
		mean = float(mean)
		stdev = float(stdev)
		if stdev == 0:
			stdev = 1
		exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
		# return sp.norm(mean, stdev).pdf(x)

	def calculate_class_probabilities(self, summaries, inputVector):
		'''
		Calculate probability of entire feature belonging to a class
		'''
		probabilities = {}
		for class_label, features in summaries.items():
			probabilities[class_label] = 1
			for i in range(len(features)):
				mean, stdev = features[i]
				x = inputVector[i]
				# Don't know what to do with nominal features
				if x.isdigit() is False: 
					x = 1
				probabilities[class_label] *= self.calculate_normal_PDF(x, mean, stdev)
		return probabilities

	def predict(self, summaries, data):
		'''
		Predict the label for the given data point
		'''
		probabilities = self.calculate_class_probabilities(summaries, data)
		best_label, best_prob = None, -1
		for class_label, probability in probabilities.items():
			if best_label is None or probability > best_prob:
				best_prob = probability
				best_label = class_label
		return best_label

	def get_predictions(self, summaries, data):
		'''
		Create a list of predictions for all data
		'''
		predictions = []
		for i in range(len(data)):
			result = self.predict(summaries, data[i])
			predictions.append(result)
		return predictions

	def get_accuracy(self, data, predictions):
		'''
		Calculate the prediction accuracy
		'''
		correct = 0
		for x in range(len(data)):
			if data[x][-1] == predictions[x]:
				correct += 1
		return (correct/float(len(data))) * 100.0


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	parser.add_argument('--split', type=float, default=0.2,
    					help="Percent of data to use for validation")
	args = parser.parse_args()

	if args.limit < 0:
		dataset = Data("kddcup.data_10_percent.gz",1-args.split)
	else:
		dataset = Data("kddcup.data_10_percent.gz",1-args.split, args.limit)

	nb = NaiveBayes(dataset)
	summaries = nb.summarize_features_by_class(dataset.train_x)
	predictions = nb.get_predictions(summaries, dataset.valid_x)
	accuracy = nb.get_accuracy(dataset.valid_x, predictions)

	print('Accuracy: {0}'.format(accuracy))
