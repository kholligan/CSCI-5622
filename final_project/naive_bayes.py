import argparse
from collections import Counter, defaultdict

import random
import math
import scipy.stats as sp
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
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

		with gzip.open(location, 'rb') as f:
			for line in f:
				x = line.decode('utf8').splitlines()
				for item in x:
					self.dataset.append(item.split(','))

		if self.limit is None:
			self.train_size = int(round(len(self.dataset)*self.split_ratio))
			self.valid_size = int(round(len(self.dataset)*(1-self.split_ratio)))
		else:
			self.train_size = int(round(self.limit))
			self.valid_size = int(round(self.limit*(1-self.split_ratio)))

		self.cont_trainx = [[] for _ in range(self.train_size)]
		self.multi_trainx = [[] for _ in range(self.train_size)]
		self.cont_validx = [[] for _ in range(self.valid_size)]
		self.multi_validx = [[] for _ in range(self.valid_size)]
		self.train_y = []
		self.valid_y = []

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

def convert_to_float(dataset):
	return_array = []
	for i, x in enumerate(dataset):
		try:
			return_array.append( [float(k) for k in x] )
		except ValueError:
			pass
	return return_array


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	parser.add_argument('--split', type=float, default=0.2,
    					help="Percent of data to use for validation")
	args = parser.parse_args()

	if args.limit < 0:
		# dataset = Data("kddcup.data_10_percent.gz",1-args.split)
		train = Data("kddcup.data_10_percent.gz", 1)
		valid = Data("corrected.gz", 1)

	else:
		dataset = Data("kddcup.data_10_percent.gz",1-args.split, args.limit)

	gnb = GaussianNB()
	mnb  = MultinomialNB()

	# # If using 10% train + corrected.gz
	train_x = convert_to_float(train.cont_trainx)
	valid_x = convert_to_float(valid.cont_trainx)
	train_y = np.array(train.train_y)
	valid_y = np.array(valid.train_y)

	train_x = np.array(train_x)
	valid_x = np.array(valid_x)

	gnb.fit(train_x, train_y)
	accuracy = gnb.score(valid_x, valid_y)

	# # For using 10% data split into train/valid set
	# train_x = convert_to_float(dataset.cont_trainx)
	# valid_x = convert_to_float(dataset.cont_validx)

	# train_x = np.array(train_x)
	# train_y = np.array(dataset.train_y)
	# valid_x = np.array(valid_x)

	# gnb.fit(train_x, train_y)
	# predictions = gnb.predict(valid_x)
	# accuracy = gnb.score(valid_x, dataset.valid_y)
	
	print('Accuracy: {0}'.format(accuracy))


	# # Failed MNB work

	# count_vect = CountVectorizer(analyzer='word')
	# mnb_train_x = np.array(dataset.multi_trainx)
	# print(mnb_train_x)
	# mnb_train_counts = count_vect.fit_transform([x for x in dataset.multi_trainx]).toarray()
	# print(dataset.multi_trainx)
	# print (word for word in (x for x in dataset.multi_trainx))
	# multi_trainx = []
	# mnb_train_counts = count_vect.fit_transform(multi_trainx)
	# print(mnb_train_x)
	# print(mnb_train_counts)

	# vec = CountVectorizer(tokenizer=lambda doc: doc, lowercase = False)
	# mnb_train_counts = vec.fit_transform(dataset.multi_trainx)

	# mnb_valid = vec.fit_transform(dataset.multi_validx)

	# print(dataset.multi_trainx)
	# print(mnb_train_counts)

	# mnb.fit(mnb_train_counts, train_y)
	# predict = mnb.predict(mnb_valid)

	# print(predict)

	# vect_repr = list(map(count_vect.fit_transform, dataset.multi_trainx))
	# # print(vect_repr)
	# mnb_train_x = np.array(vect_repr)
	# print(mnb_train_x[0])

	# mnb.fit(vect_repr, train_y)
	# mnb.fit(mnb_train_x, train_y)

