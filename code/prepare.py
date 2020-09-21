import json
import numpy as np
import os
import sys
from sklearn.model_selection import KFold
from math import log10
from tokenizer import Tokenizer
import pickle
import csv
import re
import random
random.seed(0)


class prepareData(object):
	def __init__(self, EMBEDDING_DIM, MAXLEN, TRIM_NUM):
		self.EMBEDDING_DIM=EMBEDDING_DIM
		self.MAXLEN = MAXLEN
		self.TRIM_NUM=TRIM_NUM
		self.tokenizer=Tokenizer()
	
	def cleanText(self, text):
		text=text.encode('utf-8', 'ignore')
		text = text.lower().decode('utf-8')
		#text = re.sub(r"[^a-z ]", r' ', text.decode('utf-8'))
		#print (text)
		text = re.sub(' +',' ',text)
		return text

	def _read_tsv(self, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r", encoding="utf-8") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				if sys.version_info[0] == 2:
					line = list(unicode(cell, 'utf-8') for cell in line)
				lines.append(line)
			return lines

	def tokenize_claims(self, articles):

		seqs_articles=[]
		for article in articles:
			seqs_articles.append(self.tokenizer.texts_to_sequences(article, self.MAXLEN))
	
		return seqs_articles
	def get_guided_masks(self, claim, atten):
		seqs_masks=[0 for i in range(self.MAXLEN)]
		atten_set=set(atten.split(' '))
		claim_seqs=claim.split(' ')
		for (i, word) in enumerate(claim_seqs):
			if i>=self.MAXLEN:
				break
			if word in atten_set:
				seqs_masks[i]=1
		return seqs_masks
	
	def tokenize_claims_tfidf(self, claims_list):
		docNum=len(claims_list)
		term_df = dict()
		for claim in claims_list:
			for term in set(claim.split(' ')):
				if term not in term_df:
					term_df [term]=1.0
				else:
					term_df[term]+=1.0
				
		for term in term_df:
			term_df[term] = log10(docNum/term_df[term])

		seqs_claims=[]
		for claim in claims_list:
			seqs_claim=[ 0 for j in range(self.tokenizer.num_words)]
			term_tf = dict()
			terms=claim.split(' ')
			for term in terms:
				if term not in term_tf:
					term_tf[term]=1.0
				else:
					term_tf[term]+=1.0
			docLen=len(terms)
			for term in set(terms):
				if term in self.tokenizer.word2index:
					tfidf = term_tf[term]/docLen* term_df[term]
					word_id=self.tokenizer.word2index[term]
					seqs_claim[word_id] =tfidf

			seqs_claims.append(seqs_claim)
			
		return seqs_claims
	
	def get_embeddings_index(self):
		embeddings_index = {}
		path = r'../data/glove.6B.100d.txt'
		f = open(path, 'r')
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()
		return embeddings_index


	def get_embedding_matrix(self):
		embeddings_index=self.get_embeddings_index()
		word_index=self.tokenizer.word2index
		embedding_matrix = np.zeros((len(word_index) + 1, self.EMBEDDING_DIM))
		for word, i in word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector
			else:
				embedding_matrix[i] = np.random.rand(self.EMBEDDING_DIM)
		#embedding_matrix = np.array(embedding_matrix, dtype=np.float32)
		return embedding_matrix

	def data_process(self, path_train, path_dev):
		claims_train=[]
		labels_train=[]
		#lines=self._read_tsv(path+'train-all.tsv')
		lines=self._read_tsv(path_train)
		for (i, line) in enumerate(lines):
			claims_train.append(self.cleanText(line[2]))
			labels_train.append(int(line[3]))
		
		claims_dev=[]
		labels_dev=[]
		#lines=self._read_tsv(path+'leaderboard-dev.tsv')
		lines=self._read_tsv(path_dev)
		for (i, line) in enumerate(lines):
			claims_dev.append(self.cleanText(line[2]))
			labels_dev.append(int(line[3]))
		
		self.tokenizer.fit_on_texts(claims_train+claims_dev)
		seqs_claims_train=self.tokenize_claims(claims_train)
		seqs_claims_dev=self.tokenize_claims(claims_dev)
		
		#shuffle
		np.random.seed(0)
		idx = np.arange(0, len(claims_train), 1)
		np.random.shuffle(idx)
		seqs_claims_train = [seqs_claims_train[t] for t in idx]
		labels_train = [labels_train[t] for t in idx]
		print('training set length %s'%(len(seqs_claims_train)))
		print('dev set length %s'%(len(seqs_claims_dev)))
				
		return [[seqs_claims_train, labels_train], [seqs_claims_dev, labels_dev]]
	
	def data_process_pair(self, xtrain, train_y, xvalid, valid_y, separate, sampleNum):
		train_pos, train_neg, valid_text =[], [], []
		if separate:
			self.tokenizer.fit_on_texts(xtrain)
		else:
			self.tokenizer.fit_on_texts(xtrain+xvalid)
		seqs_train=self.tokenize_claims(xtrain)
		seqs_valid=self.tokenize_claims(xvalid)
		
		pos_claims, neg_claims=[],[]
		for i, label in enumerate(train_y):
			if label==1:
				pos_claims.append(seqs_train[i])
			else:
				neg_claims.append(seqs_train[i])
		pos_len=len(pos_claims)
		neg_len=len(neg_claims)
		pos_claims=np.asarray(pos_claims)
		neg_claims=np.asarray(neg_claims)
		if pos_len<neg_len:
			neg_claims=neg_claims[0:pos_len]
		else:
			pos_claims=pos_claims[0:neg_len]
		assert(pos_claims.shape[0]==neg_claims.shape[0])
		for i in range(sampleNum[0]):
			indxs=np.random.permutation(pos_claims.shape[0])
			train_pos+=list(pos_claims)
			train_neg+=list(neg_claims[indxs])
				
		print('training set length %s'%(len(train_pos)))
		print('dev set length %s'%(len(seqs_valid)))
			
		np.random.seed(0)
		idx = np.arange(0, len(train_pos), 1)
		np.random.shuffle(idx)
		train_pos = [train_pos[t] for t in idx]
		train_neg = [train_neg[t] for t in idx]
		print(len(train_pos))
		
		return [[train_pos, train_neg], [seqs_valid, valid_y]]
	
	def data_process_bert_pair(self, path_train, path_dev, sampleNum=5):
		claims_train=[]
		labels_train=[]
		#lines=self._read_tsv(path+'train-all.tsv')
		lines=self._read_tsv(path_train)
		for (i, line) in enumerate(lines):
			claims_train.append('[CLS] '+self.cleanText(line[2])+' [SEP]')
			labels_train.append(int(line[3]))
		
		claims_dev=[]
		labels_dev=[]
		#lines=self._read_tsv(path+'leaderboard-dev.tsv')
		lines=self._read_tsv(path_dev)
		for (i, line) in enumerate(lines):
			claims_dev.append('[CLS] '+self.cleanText(line[2])+' [SEP]')
			labels_dev.append(int(line[3]))
		
		#shuffle
		#np.random.seed(0)
		#idx = np.arange(0, len(claims_train), 1)
		#np.random.shuffle(idx)
		#seqs_claims_train = [seqs_claims_train[t] for t in idx]
		#labels_train = [labels_train[t] for t in idx]
		pos_claims=[]
		neg_claims=[]
		for i, label in enumerate(labels_train):
			if label==1:
				pos_claims.append(claims_train[i])
			else:
				neg_claims.append(claims_train[i])
		pos_claims=np.asarray(pos_claims)
		neg_claims=np.asarray(neg_claims)
		pos_inds=[]
		neg_inds=[]
		neg_len=neg_claims.shape[0]
		for i in range(pos_claims.shape[0]):
			count_id=0
			while(count_id<sampleNum):
				j=random.randint(0,neg_len-1)
				pos_inds.append(i)
				neg_inds.append(j)
				count_id+=1
		
		print('training set length %s'%(len(pos_inds)))
		print('dev set length %s'%(len(claims_dev)))
				
		return [[pos_claims[pos_inds], neg_claims[neg_inds]], [claims_dev, labels_dev]]

	def data_process_bow(self, path):
		claims_train=[]
		labels_train=[]
		lines=self._read_tsv(path+'train-all.tsv')
		#lines=self._read_tsv(path+'train.tsv')
		for (i, line) in enumerate(lines):
			claims_train.append(self.cleanText(line[2]))
			labels_train.append(int(line[3]))
		
		claims_dev=[]
		labels_dev=[]
		lines=self._read_tsv(path+'leaderboard-dev.tsv')
		#lines=self._read_tsv(path+'dev.tsv')
		for (i, line) in enumerate(lines):
			claims_dev.append(self.cleanText(line[2]))
			labels_dev.append(int(line[3]))
		#initialize the voc
		all_claims=claims_train+claims_dev
		self.tokenizer.fit_on_texts(all_claims)
		self.tokenizer.trim(self.TRIM_NUM)
		
		
		##get the bow feature
		seqs_all=self.tokenize_claims_tfidf(all_claims)
		
		seqs_claims_train=seqs_all[0:len(claims_train)]
		seqs_claims_dev=seqs_all[len(claims_train):]
		
		#shuffle
		np.random.seed(0)
		idx = np.arange(0, len(claims_train), 1)
		np.random.shuffle(idx)
		seqs_claims_train = [seqs_claims_train[t] for t in idx]
		labels_train = [labels_train[t] for t in idx]
		print('training set length %s'%(len(seqs_claims_train)))
		print('dev set length %s'%(len(seqs_claims_dev)))
				
		return [[seqs_claims_train, labels_train], [seqs_claims_dev, labels_dev]]
	
	def data_process_guided(self, path):
		claims_train=[]
		labels_train=[]
		guided_train=[]
		#lines=self._read_tsv(path+'train_guided_all.tsv')
		lines=self._read_tsv(path+'train_guided.tsv')
		for (i, line) in enumerate(lines):
			claim=self.cleanText(line[2])
			claims_train.append(claim)
			labels_train.append(int(line[3]))
			attens=self.cleanText(line[4])
			guided_train.append(self.get_guided_masks(claim, attens))
			
		
		claims_dev=[]
		labels_dev=[]
		guided_dev=[]
		#lines=self._read_tsv(path+'leaderboard-dev.tsv')
		lines=self._read_tsv(path+'dev_guided.tsv')
		for (i, line) in enumerate(lines):
			claim=self.cleanText(line[2])
			claims_dev.append(claim)
			labels_dev.append(int(line[3]))
			attens=''
			guided_dev.append([])
		
		self.tokenizer.fit_on_texts(claims_train+claims_dev)
		seqs_claims_train=self.tokenize_claims(claims_train)
		seqs_claims_dev=self.tokenize_claims(claims_dev)
		
		#shuffle
		np.random.seed(0)
		idx = np.arange(0, len(claims_train), 1)
		np.random.shuffle(idx)
		seqs_claims_train = [seqs_claims_train[t] for t in idx]
		labels_train = [labels_train[t] for t in idx]
		guided_train = [guided_train[t] for t in idx]
		print('training set length %s'%(len(seqs_claims_train)))
		print('dev set length %s'%(len(seqs_claims_dev)))
				
		return [[seqs_claims_train, labels_train, guided_train], [seqs_claims_dev, labels_dev, guided_dev]]
	
	def data_process_partly_guided(self, path):
		claims_train=[]
		labels_train=[]
		lines=self._read_tsv(path+'train_guided.tsv')
		for (i, line) in enumerate(lines):
			claim=self.cleanText(line[2])	
			claim_words=claim.split(' ')
			claim=' '.join(claim_words[0:5])
			label=int(line[3])
			labels_train.append(label)
			if label==1:
				atten=self.cleanText(line[4])
				if atten!='':
					claim=atten
			claims_train.append(claim)
			
		
		claims_dev=[]
		labels_dev=[]
		lines=self._read_tsv(path+'dev_guided.tsv')
		for (i, line) in enumerate(lines):
			claim=self.cleanText(line[2])	
			claim_words=claim.split(' ')
			claim=' '.join(claim_words[0:5])
			label=int(line[3])
			labels_dev.append(label)
			if label==1:
				atten=self.cleanText(line[4])
				if atten!='':
					claim=atten
			claims_dev.append(claim)

		
		self.tokenizer.fit_on_texts(claims_train+claims_dev)
		seqs_claims_train=self.tokenize_claims(claims_train)
		seqs_claims_dev=self.tokenize_claims(claims_dev)
		
		#shuffle
		np.random.seed(0)
		idx = np.arange(0, len(claims_train), 1)
		np.random.shuffle(idx)
		seqs_claims_train = [seqs_claims_train[t] for t in idx]
		labels_train = [labels_train[t] for t in idx]
		print('training set length %s'%(len(seqs_claims_train)))
		print('dev set length %s'%(len(seqs_claims_dev)))
				
		return [[seqs_claims_train, labels_train], [seqs_claims_dev, labels_dev]]
#path = '/GW/D5data-11/lwang/rlfake/propaganda/data/'
#prepare=prepareData(50,128, 0)
#[train_sets, dev_sets] = prepare.data_process(path)
