import os

def split_data(X, y, ratio):
	split_index = int(len(X)*ratio)
	train_pair = [(X[i], y[i]) for i in range(split_index)]
	val_pair = [(X[i], y[i]) for i in range(split_index, len(X))]
	return train_pair, val_pair