import argparse
from collections import Counter, defaultdict

import random
import numpy
import gzip
import re

class Data:
	"""
	Class to hold the data set.
	"""

	def __init__(self, location, split_ratio):

		self.dataset = []
		self.split_ratio = split_ratio
		self.train_x = []
		self.train_y = []
		self.valid_x = []
		self.valid_y = []
		i = 0
		with gzip.open(location, 'rb') as f:
			for line in f:
				x = line.decode('utf8').splitlines()
				for item in x:
					self.dataset.append(item.split(','))
		self.splitDataset()

	def splitDataset(self):
		train_size = int(len(self.dataset)*self.split_ratio)
		# print(len(self.dataset), self.split_ratio, train_size)
		copy = self.dataset
		print(len(copy))
		# Randomly pull items from the data set for the training set
		while len(self.train_x) < train_size:
			index = random.randrange(len(copy))
			item = copy.pop(index)
			self.train_y.append(item[-1])
			self.train_x.append(item[:-1])

		# Put the remaining items in the validation set
		for item in copy:
			self.valid_x.append(item[-1])
			self.valid_y.append(item[:-1])

class NaiveBayes():
	"""
	Class to compute naive bayes probabilities
	"""

	def __init__(self):
		pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument('--split', type=float, default=0.2,
    					help="Percent of data to use for validation")
    args = parser.parse_args()

    data = Data("kddcup.data_10_percent.gz",1-args.split)
