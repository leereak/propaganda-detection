from __future__ import print_function

import numpy as np
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

sys.path.append('../code/')
from prepare import prepareData
from my_dataset import myDataset, myDataset_pair
from common import *


torch.set_default_tensor_type('torch.DoubleTensor')

EMBEDDING_DIM = 100 #embedding dimension
MAXLEN_CLAIM = 128
HIDDEN_DIM=100
BATCH_SIZE=100
TRIM_NUM=0
ITER_NUM=20
LR=5e-5
dataset_org='hnds'
fold_num=5
separate=True
sampleNum=[1]
basePath='../data/'
dataset_train=basePath+dataset_org


args={'separate':separate, 'dataset':dataset_org, 'EMBEDDING_DIM':EMBEDDING_DIM, 'MAXLEN_CLAIM':MAXLEN_CLAIM, 
'BATCH_SIZE':BATCH_SIZE, 'ITER_NUM':ITER_NUM, 'TRIM_NUM':TRIM_NUM, 'LR':LR, 'HIDDEN_DIM':HIDDEN_DIM, 'fold_num':fold_num, 'basePath':basePath, 'SampleNum':sampleNum}
print (args)


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

def My_loss_logit(pos, neg):
	#####sign loss
	#differ_sign=torch.sign(1+torch.sign(neg-pos))
	
	####gap loss
	differ=torch.log(1+torch.exp(neg-pos))
	
	return torch.mean(differ,0)
	
def My_loss_me(pos, neg):
	
	####gap loss
	differ=1+(neg-pos)
	
	return torch.mean(differ,0)

def My_loss_mte(pos, neg):
	###threshold loss
	temp_differ=(1+(neg-pos))/2
	temp_zero=torch.zeros_like(temp_differ)
	differ=torch.where(temp_differ<0.4, temp_zero, temp_differ)
	
	return torch.mean(differ,0)

def My_loss_ct(pos, neg):
	#####sign loss
	differ=torch.sign(1+torch.sign(neg-pos))
	return torch.mean(differ,0)

class claim_classifier():
	def __init__(self):
		self.net = biLSTM_module().to(device)
		print(self.net)
		
		#self.criterion = nn.MSELoss()
		self.criterion = My_loss_ct
		#self.criterion = nn.BCELoss()
		#self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
		self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
	
	
	def train(self, trainloader, devloader):
		for epoch in range(ITER_NUM):  # loop over the dataset multiple times
			running_loss = 0.0
			running_loss_all=0.0
			for i, data in enumerate(trainloader,0):
				self.net.train()
				pos, neg = data
				pos, neg=  pos.to(device), neg.to(device)
				self.optimizer.zero_grad()
				# forward + backward + optimize
				outputs_pos = self.net(pos)
				outputs_neg = self.net(neg)
				loss = self.criterion(outputs_pos, outputs_neg)
				loss.backward()
				self.optimizer.step()
				
				running_loss += loss.item()
				running_loss_all += loss.item()
				if i % 50 == 49:    # print every 100 mini-batches
					print('Batch:%d loss:%.4f' %(i + 1, running_loss / 50))
					running_loss = 0.0
			#train_f1 = self.predict(trainloader, False)
			dev_f1 = self.predict( devloader, False)
			
			print('Epoch:%d \tloss:%.4f \tdev_acc:%s' %(epoch + 1, running_loss_all / len(trainloader.dataset)*BATCH_SIZE, dev_f1))
			

		print('Finished Training')
		
	def predict(self, testloader, out_to_file):
		
		all_labels=np.array([])
		all_preds=np.array([])
		
		total = 0.0
		for i, data in enumerate(testloader,0):
			claims, labels = data
			claims, labels =  claims.to(device), labels
			
			self.net.eval()
			with torch.no_grad():
				outputs = self.net(claims )
			
			#probs, _= torch.max(outputs, 1)
			preds=outputs.cpu().numpy()
			all_labels=np.concatenate((all_labels, labels.numpy()))
			all_preds=np.concatenate((all_preds, preds[:,0]))
			
		if out_to_file:
			with open('lstm-eval-results-dev.txt', 'w') as writeF:
				for label in all_preds:
					writeF.write('%s\n'%label)
				writeF.close()
				
		predicted=[1 if x>=0.5 else 0 for x in all_preds]
		results= precision_recall_fscore_support(y_true=all_labels, y_pred= np.asarray(predicted), average='binary', pos_label=1)
		average_1=np.mean(all_preds[all_labels==1])
		average_0=np.mean(all_preds[all_labels==0])
		var_1=np.var(all_preds[all_labels==1])
		var_0=np.var(all_preds[all_labels==0])
		print('average score and var of class 1 and 0 ', average_1, average_0, average_1-average_0, var_1, var_0)
			
		return results[0:3]

	
	def save_model(self, PATH):
		torch.save(self.net.state_dict(), PATH)
	
	def load_model(self, PATH):
		self.net.load_state_dict(torch.load(PATH))

class biLSTM_module(nn.Module):

	def __init__(self):
		super(biLSTM_module, self).__init__()
		
		self.embedding=nn.Embedding.from_pretrained(pretrained_embeddings)
		
		self.lstm_claim = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
		
		self.fc1 = nn.Linear(HIDDEN_DIM*2, 50)
		
		self.fc2 = nn.Linear(50, 1)

	def forward(self, claims):
		
		embeds_claim = self.embedding(claims)
		_, (lstm_out_claim, _) = self.lstm_claim(embeds_claim)
		x=torch.cat((lstm_out_claim[0],lstm_out_claim[1]), 1)
		
		x=F.relu(self.fc1(x))
		output = torch.sigmoid(self.fc2(x))
		
		return output



print ('train dataset: ', dataset_train)
xtrain, train_y=load_dataset_text(dataset_train, False)


dataset_name='sp-a'
dataset_name=basePath+dataset_name
print ('test dataset', dataset_name)
xvalid, valid_y=load_dataset_text(dataset_name, False)

prepare=prepareData(EMBEDDING_DIM, MAXLEN_CLAIM, TRIM_NUM)
[train_sets, dev_sets] = prepare.data_process_pair(xtrain, train_y, xvalid, valid_y, separate, sampleNum)
pretrained_embeddings=torch.tensor(prepare.get_embedding_matrix()).to(device)


dataset=myDataset_pair(train_sets[0],train_sets[1])
trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,  shuffle=False)

dataset=myDataset(dev_sets[0], dev_sets[1])
devloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


classifier=claim_classifier()
classifier.train(trainloader, devloader)
test_datasets=['sp-a', 'sp-s','ira','hnda','hnds']
test_datasets.remove(dataset_org)
for testset in test_datasets:
	dataset_name=basePath+testset
	print ('test dataset', dataset_name)
	xvalid, valid_y=load_dataset_text(dataset_name, False)
	valid_seqs=prepare.tokenize_claims(list(xvalid))
	dataset=myDataset(valid_seqs, valid_y)
	devloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
	dev_f1=classifier.predict(devloader, False)
	res_str=' &'.join(["%.3f" %x for x in dev_f1])
	print ("Method: LSTM feature performance : &{} ".format(res_str))




