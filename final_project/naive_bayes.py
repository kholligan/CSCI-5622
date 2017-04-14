import argparse
from collections import Counter, defaultdict

import random
import math
import scipy.stats as sp
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
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

		self.header_type = ['continuous', 'symbolic', 'symbolic', 'symbolic', 'continuous',
				'continuous', 'symbolic','continuous', 'continuous', 'continuous', 
				'continuous', 'symbolic','continuous', 'continuous', 'continuous', 
				'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 
				'symbolic', 'symbolic',	'continuous', 'continuous', 'continuous', 
				'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 
				'continuous', 'continuous', 'continuous', 'continuous', 'continuous',
				'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 
				'continuous', 'class']
		self.limit = limit

		if self.limit is None:
			self.train_size = int(len(self.dataset)*self.split_ratio)
			self.valid_size = int(len(self.dataset)*(1-self.split_ratio))
		else:
			self.train_size = int(round(self.limit))
			self.valid_size = int(round(self.limit*(1-self.split_ratio)))

		self.cont_trainx = [[] for _ in range(self.train_size)]
		self.multi_trainx = [[] for _ in range(self.train_size)]
		self.cont_validx = [[] for _ in range(self.valid_size)]
		self.multi_validx = [[] for _ in range(self.valid_size)]
		self.train_y = []
		self.valid_y = []

		with gzip.open(location, 'rb') as f:
			for line in f:
				x = line.decode('utf8').splitlines()
				for item in x:
					self.dataset.append(item.split(','))
		self.splitDataset()

	def splitDataset(self):
		# print(len(self.dataset), self.split_ratio, train_size)
		copy = self.dataset
		# Randomly pull items from the data set for the training set
		data_index = 0
		while data_index < self.train_size:
			index = random.randrange(len(copy))
			item = copy.pop(index)
			for i in range(len(item)):
				if self.header_type[i] == 'continuous':
					self.cont_trainx[data_index].append(item[i])
				elif self.header_type[i] == 'symbolic':
					self.multi_trainx[data_index].append(item[i])
				else:
					self.train_y.append(item[i])
			data_index += 1

		data_index = 0
		# Put the remaining items in the validation set
		while data_index < self.valid_size:
			index = random.randrange(len(copy))
			item = copy.pop()
			for i in range(len(item)):
				if self.header_type[i] == 'continuous':
					self.cont_validx[data_index].append(item[i])
				elif self.header_type[i] == 'symbolic':
					self.multi_validx[data_index].append(item[i])
				else:
					self.valid_y.append(item[i])
			data_index += 1

def convert_to_float(list):
	pass

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

	gnb = GaussianNB()
	mnb  = MultinomialNB()

	# Convert values to floats for training and validation data
	train_x = []
	for i, x in enumerate(dataset.cont_trainx):
		try:
			train_x.append( [float(k) for k in x] )
		except ValueError:
			pass

	valid_x = []
	for i, x in enumerate(dataset.cont_validx):
		try:
			valid_x.append( [float(k) for k in x] )
		except ValueError:
			pass

	# print(dataset.cont_trainx)
	# train_x = [int(i) for i in dataset.cont_trainx]
	# print(train_x)
	train_x = np.array(train_x)

	# X = np.array([np.array(xi) for xi in train_x])
	train_y = np.array(dataset.train_y)

	gnb.fit(train_x, train_y)
	valid_x = np.array(valid_x)
	# print (dataset.cont_validx)
	# mnb.fit(dataset.multi_trainx, dataset.train_y)
	prediction = gnb.predict(valid_x)

