from utils.loader import load, build_data
from utils.split import split_data
from models.model import CBOW
import pdb
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse
import os
import numpy as np
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--num_epochs', type=int, default= 2)
argparser.add_argument('--hidden_size', type=int, default=1000)
argparser.add_argument('--learning_rate', type=float, default= 0.001)
argparser.add_argument('--batch_size', type = int, default= 512)
args = argparser.parse_args()



def train(use_gpu,X, y, model, criterion, optimizer,num_epochs, batch_size):
	model.train()
	train_size = len(X)
	for epoch in range(num_epochs):
		correct =0
		for i in range(0, train_size, batch_size):
			num_samples = min(batch_size, train_size - i)
			data = torch.from_numpy(X[i:i+num_samples])
			targets = torch.from_numpy(y[i:i+num_samples])
			
			if use_gpu:
				data, targets = data.cuda(), targets.cuda()

			data, targets = Variable(data), Variable(targets)
			# pdb.set_trace()
			model.zero_grad()
			output = model(data)
			loss = criterion(output, targets)
			loss.backward()
			optimizer.step()
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			# pdb.set_trace()
			correct += pred.eq(targets.data).cpu().sum()
			accuracy = float(correct)/train_size*100
			print('Accuracy: %.4f'%(accuracy))
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'%(epoch+1, num_epochs, i+1,\
			len(data)//train_size, loss.data[0]))

	return model


		

if __name__ == '__main__':

	data = load(feat="vocab")
	num_epochs = args.num_epochs
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	embedding_size = args.hidden_size
	context_size = 2

	def save(model_ft):
		save_filename = args.filename
		torch.save(model_ft, save_filename)
		print('Saved as %s' % save_filename)

	if data is None:
		print("loading....")
		data = build_data(10000)
	word_ids = data["words"]
	vocab = data["vocabulary"]
	vocab_size = len(vocab)

	# preparing data with fixed contetext size
	X, y = [], [] 
	for i in range(2, len(word_ids) - 2):
		X.append([word_ids[i - 2], word_ids[i - 1], word_ids[i + 1], word_ids[i + 2]])
		y.append(word_ids[i])

	# splitting the data in a 80-20 ratio
	train_pair, val_pair = split_data(X, y, 0.8)
	X_train, y_train = [], []
	for x, label in train_pair:
		X_train.append(x)
		y_train.append(label)

	# Load model
	print("loading model")
	use_gpu = torch.cuda.is_available()
	model = CBOW(vocab_size, embedding_size, context_size)
	if use_gpu:
		print("using GPU")
		model = model.cuda()

	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	print("training...")
	model = train(use_gpu, np.array(X_train), np.array(y_train), model, criterion, optimizer,num_epochs, batch_size)
	save(model)



