import random
from random import randrange
import pandas as pd


def feat_num_try_half(max_index):
	v = []
	while max_index > 1:
		v.append(max_index)
		max_index //= 2
	return v


def feat_num_try(f_tuple):
	for i in range(len(f_tuple)):
		if f_tuple[i][1] < 1e-20:
			i = i-1
			break
	return feat_num_try_half(i+1)[:8]


def random_shuffle(label, sample):
	random.seed(1)
	size = len(label)
	for i in range(size):
		ri = randrange(0, size-i)
		tmp = label[ri]
		label[ri] = label[size-i-1]
		label[size-i-1] = tmp
		tmp = sample[ri]
		sample[ri] = sample[size-i-1]
		sample[size-i-1] = tmp


def value_cmpf(x):
	return (-x[1])


def cal_feat_imp(label, sample):
	score_dict = cal_Fscore(label, sample)

	score_tuples = list(score_dict.items())
	score_tuples.sort(key=value_cmpf)

	feat_v = score_tuples
	for i in range(len(feat_v)):
		feat_v[i] = score_tuples[i][0]
	return score_dict, feat_v


def fscore_main(data, filename):
	train_label, train_sample, max_index, features = read_data_csv(data)
	# Randomly shuffle data
	random_shuffle(train_label, train_sample)
	whole_fsc_dict, whole_imp_v = cal_feat_imp(train_label, train_sample)
	# output
	fscore_data = pd.DataFrame(columns=["Feature", "Fscore"])
	for i in whole_fsc_dict.keys():
		fscore_data.loc[i-1] = [features[i-1], whole_fsc_dict[i]]
	fscore_data = fscore_data.sort_values('Fscore', ascending=False)
	fscore_data.to_csv("{}_fscore.csv".format(filename), index=False)
	train_data = data.reindex(['Label']+list(fscore_data['Feature']), axis=1)
	train_data.to_csv("{}_fscore_data.csv".format(filename), index=False)
	return train_data


def cal_Fscore(labels, samples):

	data_num=float(len(samples))
	p_num = {}
	sum_f = []
	sum_l_f = {}
	sumq_l_f = {}
	F={}
	max_idx = -1
	for p in range(len(samples)):
		label = labels[p]
		point = samples[p]

		if label in p_num: p_num[label] += 1
		else: p_num[label] = 1

		for f in point.keys():
			if f > max_idx:
				max_idx=f

	sum_f = [0 for i in range(max_idx)]
	for la in p_num.keys():
		sum_l_f[la] = [0 for i in range(max_idx)]
		sumq_l_f[la] = [0 for i in range(max_idx)]

	for p in range(len(samples)):
		point=samples[p]
		label=labels[p]
		for tuple in point.items():
			f = tuple[0]-1
			v = tuple[1]
			sum_f[f] += v
			sum_l_f[label][f] += v
			sumq_l_f[label][f] += v**2

	eps = 1e-12
	for f in range(max_idx):
		SB = 0
		for la in p_num.keys():
			SB += (p_num[la] * (sum_l_f[la][f]/p_num[la] - sum_f[f]/data_num)**2 )

		SW = eps
		for la in p_num.keys():
			SW += (sumq_l_f[la][f] - (sum_l_f[la][f]**2)/p_num[la]) 

		F[f+1] = SB / SW

	return F


# read csv
def read_data_csv(data):
	max_index = data.shape[1]-1
	labels = list(data.iloc[:, 0])
	samples = []
	for i in range(data.shape[0]):
		sample = {}
		for j in range(1, data.shape[1]):
			sample[j] = data.iloc[i, j]
		samples.append(sample)
	features = list(data.columns[1:])
	return labels, samples, max_index, features
