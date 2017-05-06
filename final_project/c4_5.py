from __future__ import division
import argparse
import gzip,random
#from sklearn.naive_bayes import convert_to_float
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import itertools

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
		a = np.ascontiguousarray(self.dataset)
		unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
		uniques=unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
		self.dataset=uniques.tolist()
		print(len(self.dataset))
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
		self.train_x = []
		self.valid_x = []
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
			self.train_x.append(item[:-1])

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
			self.valid_x.append(item[:-1])

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

	args.limit = 100
	if args.limit < 0:
		dataset = Data("corrected.gz",1-args.split)
		# train = Data("kddcup.data_10_percent.gz", 1)
		# valid = Data("corrected.gz", 1)

	else:
		dataset = Data("kddcup.data_10_percent.gz", 1 - args.split, args.limit)


	# convert continuous data to floats and store it in np arrays
	cts_train_x = np.array(convert_to_float(dataset.cont_trainx), dtype='int')
	cts_valid_x = np.array(convert_to_float(dataset.cont_validx), dtype='int')


	# CALCULATING VARIANCE OF FEATURES AND REMOVING USELESS ONES

	selector = VarianceThreshold(0.01)

	cts_train_x=selector.fit_transform(cts_train_x)
	cts_valid_x=selector.fit_transform(cts_valid_x)

	#NORMALIZATION OF FEATURES
	cts_train_x= preprocessing.normalize(cts_train_x, norm='l2')
	cts_valid_x= preprocessing.normalize(cts_valid_x, norm='l2')


	# encode y labels
	le = LabelEncoder()
	labels = le.fit_transform(np.hstack((dataset.train_y, dataset.valid_y)))
	train_y = labels[0:len(dataset.train_y)]
	valid_y = labels[len(dataset.train_y):]

	# break categorical xtrain data into 2 sets: strings and numbers
	cat_str = np.asarray(dataset.multi_trainx)[:, 0:3]

	cat_num = np.asarray(dataset.multi_trainx)[:, 3:]
	cat_num = cat_num.astype(np.float)
	# encode the columns of string data into numbers
	cat_enc = np.zeros(np.shape(cat_str))
	for k in range(np.shape(cat_str)[1]):
		lab = LabelEncoder()
		cat_enc[:, k] = lab.fit_transform(cat_str[:, k])

	# recombine the categorical xtrain data
	cat_train_x = np.zeros(np.shape(dataset.multi_trainx))
	cat_train_x[:, 0:3] = cat_enc
	cat_train_x[:, 3:] = cat_num

	# break categorical xvalid data into 2 sets: strings and numbers
	cs = np.asarray(dataset.multi_validx)[:, 0:3]
	cn = np.asarray(dataset.multi_validx)[:, 3:]
	cn = cn.astype(np.float)
	# encode the columns into numbers
	ce = np.zeros(np.shape(cs))
	for kk in range(np.shape(cs)[1]):
		ll = LabelEncoder()
		ce[:, kk] = ll.fit_transform(cs[:, kk])

	# recombine categorical xvalid data
	cat_valid_x = np.zeros(np.shape(dataset.multi_validx))
	cat_valid_x[:, 0:3] = ce
	cat_valid_x[:, 3:] = cn

	#cat_train_x=selector.fit_transform(cat_train_x)
	#cat_valid_x=selector.fit_transform(cat_valid_x)

	cat_train_x= preprocessing.normalize(cat_train_x, norm='l2')
	cat_valid_x= preprocessing.normalize(cat_valid_x, norm='l2')
	# initialize models
	mod_cts = tree.DecisionTreeRegressor()
	mod_cat = tree.DecisionTreeClassifier(criterion='entropy')

	# use pipeline to combine continuous and categorical classifiers
	pipe = Pipeline([('continuous', mod_cts), ('categorical', mod_cat)])
	pipe.fit(np.hstack((cat_train_x, cts_train_x)), train_y)
	train_acc = pipe.score(np.hstack((cat_train_x, cts_train_x)), train_y)
	print 'train accuracy: %f' % train_acc
	pred = pipe.predict(np.hstack((cat_valid_x, cts_valid_x)))
	acc = pipe.score(np.hstack((cat_valid_x, cts_valid_x)), valid_y)
	print 'test accuracy: %f' % acc
