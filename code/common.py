from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
import re
import sys
import os
sys.path.append('/GW/D5data-7/lwang/tools/dictionary/lexica/')
from util import liwc, extract
from scipy.sparse import csr_matrix, diags
from sklearn import metrics
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
import pandas
import pickle
import json
import random

np.random.seed(0)
random.seed(0)


###clean the text to keep only words
def cleanText(text):
	text=text.encode('utf-8', 'ignore')
	text = text.lower().decode('utf-8')
	text = re.sub(r"[^a-z ]", r' ', text)
	text = re.sub(' +',' ',text)
	return text

###remove the proper nouns according to the POS
def removePOS(text, pos_set={'NNP', 'NNPS'}):
	tokens=nltk.word_tokenize(text)
	token_pos=nltk.pos_tag(tokens)
	result=[]
	for (word, pos) in token_pos:
		if pos not in pos_set:
			result.append(word)
	
	return ' '.join(result)

###extract LIWC features
def liwc_feature(lex, cats_to_feature, docs):
	row = []
	col = []
	data = []
	for i, doc in enumerate(docs):
		countings=extract(lex, doc, percentage=True)
		for item in countings:
			row.append(i)
			col.append(cats_to_feature[item])
			data.append(countings[item])
		if i%5000==0:
			print(i)
	result=csr_matrix((data, (row, col)))
	return result
	
### train and test a classifier
def train_model(classifier, feature_vector_train, label):
	print(classifier.get_params())
	# fit the training dataset on the classifier
	classifier.fit(feature_vector_train, label)
	# predict the labels on validation dataset
	return classifier
	

def predict_model(classifier, feature_vector_valid, valid_y):
	
	predictions = classifier.predict(feature_vector_valid)
	results= metrics.precision_recall_fscore_support(y_true=valid_y, y_pred=predictions, average='binary', pos_label=1)
	#return {
		#"precision": results[0],
		#"recall": results[1],
		#"f1": results[2],
		#"sup": results[3],
	#}
	return results[0:3]

###feature analysis for LR multi class
def feature_extract_lr(classifier, feature_vec, topn):
	#print(classifier.coef_)
	for i in range(classifier.coef_.shape[0]):
		class_prob_sorted = (-classifier.coef_[i, :]).argsort()
		print('Class %s :'%i,'\t'.join(np.take(feature_vec, class_prob_sorted[:topn])))

###feature analysis for  LR binary
def feature_extract_lr_2(classifier, feature_vec, topn):
	#print (classifier.coef_[0, (classifier.coef_[0, :]).argsort()][:topn])
	#print (classifier.coef_[0, (-classifier.coef_[0, :]).argsort()][:topn])
	neg_class_prob_sorted = (classifier.coef_[0, :]).argsort() ###increasing order
	pos_class_prob_sorted = (-classifier.coef_[0, :]).argsort() ###decresing order
	print('Class 0: ', '\t'.join(np.take(feature_vec, neg_class_prob_sorted[:topn])))
	print('Class 1: ', '\t'.join(np.take(feature_vec, pos_class_prob_sorted[:topn])))

###feature analysis for SVM
def feature_extract_svm(classifier, feature_vec, topn):
	for i in range(classifier.coef_.shape[0]):
		class_prob_sorted = (-classifier.coef_[i, :]).argsort()
		print('Class %s :'%i,'\t'.join(np.take(feature_vec, class_prob_sorted[:topn])))

###load the data and extract the features
def load_extract_data(dataset, if_remove_pos, if_scale, fold_num):
	pikPath=dataset
	if if_remove_pos:
		pikPath+='-pos.pik'
	else:
		pikPath+='.pik'
	labels, texts = [], []
	cats_to_feature={}
	splits_count, splits_tfidf, splits_ngram, splits_liwc=[],[],[],[]
	count_vect, tfidf_vect, tfidf_vect_ngram, x_liwc=[],[],[],[]

	###################### load the datasets
	if os.path.exists(pikPath):
		with open(pikPath,'rb') as f:
			[splits_count, splits_tfidf, splits_ngram, splits_liwc, count_vect, tfidf_vect, tfidf_vect_ngram, cats_to_feature]  = pickle.load(f)
			print('Data is loaded from  %s file'%pikPath)
	else:
		texts, labels=load_dataset_text(dataset, if_remove_pos)

		# creat LIWC features
		#lex, id_to_cats = liwc.parse_liwc("2015")
		#for i, item in enumerate(id_to_cats):
			#cats_to_feature[id_to_cats[item]]=i
		#x_liwc=liwc_feature(lex, cats_to_feature, texts)
		x_liwc, cats_to_feature=load_liwc(dataset, if_remove_pos)
		print('LIWC feature finish')

		# create a count vectorizer object 
		count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
		count_vect.fit(texts)
		x_count =  count_vect.transform(texts)
		print('WC feature finish')


		# word level tf-idf
		tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
		tfidf_vect.fit(texts)
		x_tfidf =  tfidf_vect.transform(texts)
		print('TI-WORD feature finish')

		# ngram level tf-idf 
		tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
		tfidf_vect_ngram.fit(texts)
		x_ngram =  tfidf_vect_ngram.transform(texts)
		print('TI-GRAM feature finish')

		## characters level tf-idf
		#tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
		#tfidf_vect_ngram_chars.fit(trainDF['text'])
		#xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
		#xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
		
		if if_scale:
			transformer1 = MaxAbsScaler().fit(x_liwc)
			x_liwc=transformer1.transform(x_liwc)
			
			transformer2 = MaxAbsScaler().fit(x_count)
			x_count=transformer2.transform(x_count)
			
			transformer3 = MaxAbsScaler().fit(x_tfidf)
			x_tfidf=transformer3.transform(x_tfidf)
			
			transformer4 = MaxAbsScaler().fit(x_ngram)
			x_ngram=transformer4.transform(x_ngram)

		#5 fold training-testing splits
		labels=np.asarray(labels)
		skf = StratifiedKFold(n_splits=fold_num, random_state=0, shuffle=True)
		for train_index, test_index in skf.split(texts, labels):
			splits_count.append([x_count[train_index,], pandas.Series(labels[train_index]), x_count[test_index,], pandas.Series(labels[test_index])])
			splits_tfidf.append([x_tfidf[train_index,], pandas.Series(labels[train_index]), x_tfidf[test_index,], pandas.Series(labels[test_index])])
			splits_ngram.append([x_ngram[train_index,], pandas.Series(labels[train_index]), x_ngram[test_index,], pandas.Series(labels[test_index])])
			splits_liwc.append([x_liwc[train_index,], pandas.Series(labels[train_index]), x_liwc[test_index,], pandas.Series(labels[test_index])])
			
		# save pik file
		with open(pikPath, 'wb') as outfile:
			pickle.dump([splits_count, splits_tfidf, splits_ngram, splits_liwc, count_vect, tfidf_vect, tfidf_vect_ngram, cats_to_feature], outfile)
		print('features are written to %s file'%pikPath)
		
	return splits_count, splits_tfidf, splits_ngram, splits_liwc, count_vect, tfidf_vect, tfidf_vect_ngram, cats_to_feature


###load the data and extract the features
def load_extract_data_cross(dataset_train, dataset_test, if_remove_pos, if_scale):
	
	count_vect, tfidf_vect, tfidf_vect_ngram, cats_to_feature=[],[],[], {}

	###################### load the datasets
	texts_train, labels_train=load_dataset_text(dataset_train, if_remove_pos)
	texts_test, labels_test=load_dataset_text(dataset_test, if_remove_pos)

	x_liwc_train, cats_to_feature=load_liwc(dataset_train, if_remove_pos)
	x_liwc_test, _=load_liwc(dataset_test, if_remove_pos)
	# create a count vectorizer object 
	count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
	count_vect.fit(texts_train)
	x_count_train =  count_vect.transform(texts_train)
	x_count_test =  count_vect.transform(texts_test)
	print('WC feature finish')


	# word level tf-idf
	tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
	tfidf_vect.fit(texts_train)
	x_tfidf_train =  tfidf_vect.transform(texts_train)
	x_tfidf_test =  tfidf_vect.transform(texts_test)
	print('TI-WORD feature finish')

	# ngram level tf-idf 
	tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
	tfidf_vect_ngram.fit(texts_train)
	x_ngram_train =  tfidf_vect_ngram.transform(texts_train)
	x_ngram_test =  tfidf_vect_ngram.transform(texts_test)
	print('TI-GRAM feature finish')

	## characters level tf-idf
	#tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
	#tfidf_vect_ngram_chars.fit(trainDF['text'])
	#xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
	#xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
	
	if if_scale:
		transformer1 = MaxAbsScaler().fit(x_liwc_train)
		x_liwc_train=transformer1.transform(x_liwc_train)
		x_liwc_test=transformer1.transform(x_liwc_test)
		
		transformer2 = MaxAbsScaler().fit(x_count_train)
		x_count_train=transformer2.transform(x_count_train)
		x_count_test=transformer2.transform(x_count_test)
		
		transformer3 = MaxAbsScaler().fit(x_tfidf_train)
		x_tfidf_train=transformer3.transform(x_tfidf_train)
		x_tfidf_test=transformer3.transform(x_tfidf_test)
		
		transformer4 = MaxAbsScaler().fit(x_ngram_train)
		x_ngram_train=transformer4.transform(x_ngram_train)
		x_ngram_test=transformer4.transform(x_ngram_test)

	#5 fold training-testing splits
	return [x_count_train, x_tfidf_train, x_ngram_train, x_liwc_train, labels_train, x_count_test, x_tfidf_test, x_ngram_test, x_liwc_test, labels_test, count_vect, tfidf_vect, tfidf_vect_ngram, cats_to_feature]

###load the data and extract the features
def load_extract_data_cross_combine(dataset_train, dataset_test, if_remove_pos, if_scale):
	
	count_vect, tfidf_vect, tfidf_vect_ngram, cats_to_feature=[],[],[], {}
	texts_train, labels_train=[],[]
	x_liwc_train=[]
	###################### load the datasets
	for train_file in dataset_train:
		texts_part, labels_part=load_dataset_text(train_file, if_remove_pos)
		texts_train+=texts_part
		labels_train+=labels_part
		x_liwc_part, cats_to_feature=load_liwc(train_file, if_remove_pos)
		x_liwc_train+=list(x_liwc_part.toarray())
	x_liwc_train=np.asarray(x_liwc_train)
	texts_test, labels_test=load_dataset_text(dataset_test, if_remove_pos)
	x_liwc_test, _=load_liwc(dataset_test, if_remove_pos)
	# create a count vectorizer object 
	count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
	count_vect.fit(texts_train)
	x_count_train =  count_vect.transform(texts_train)
	x_count_test =  count_vect.transform(texts_test)
	print('WC feature finish')


	# word level tf-idf
	tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
	tfidf_vect.fit(texts_train)
	x_tfidf_train =  tfidf_vect.transform(texts_train)
	x_tfidf_test =  tfidf_vect.transform(texts_test)
	print('TI-WORD feature finish')

	# ngram level tf-idf 
	tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
	tfidf_vect_ngram.fit(texts_train)
	x_ngram_train =  tfidf_vect_ngram.transform(texts_train)
	x_ngram_test =  tfidf_vect_ngram.transform(texts_test)
	print('TI-GRAM feature finish')

	## characters level tf-idf
	#tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
	#tfidf_vect_ngram_chars.fit(trainDF['text'])
	#xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
	#xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
	
	if if_scale:
		transformer1 = MaxAbsScaler().fit(x_liwc_train)
		x_liwc_train=transformer1.transform(x_liwc_train)
		x_liwc_test=transformer1.transform(x_liwc_test)
		
		transformer2 = MaxAbsScaler().fit(x_count_train)
		x_count_train=transformer2.transform(x_count_train)
		x_count_test=transformer2.transform(x_count_test)
		
		transformer3 = MaxAbsScaler().fit(x_tfidf_train)
		x_tfidf_train=transformer3.transform(x_tfidf_train)
		x_tfidf_test=transformer3.transform(x_tfidf_test)
		
		transformer4 = MaxAbsScaler().fit(x_ngram_train)
		x_ngram_train=transformer4.transform(x_ngram_train)
		x_ngram_test=transformer4.transform(x_ngram_test)

	#5 fold training-testing splits
	return [x_count_train, x_tfidf_train, x_ngram_train, x_liwc_train, labels_train, x_count_test, x_tfidf_test, x_ngram_test, x_liwc_test, labels_test, count_vect, tfidf_vect, tfidf_vect_ngram, cats_to_feature]

def load_liwc(dataset, if_remove_pos):
	x_liwc, cats_to_feature=[], {}
	pikPath=dataset
	if if_remove_pos:
		pikPath+='-liwc-pos.pik'
	else:
		pikPath+='-liwc.pik'
	if os.path.exists(pikPath):
		with open(pikPath,'rb') as f:
			[x_liwc, cats_to_feature]  = pickle.load(f)
			print('Data is loaded from  %s file'%pikPath)
	else:
		texts, labels=load_dataset_text(dataset, if_remove_pos)
		# creat LIWC features
		lex, id_to_cats = liwc.parse_liwc("2015")
		for i, item in enumerate(id_to_cats):
			cats_to_feature[id_to_cats[item]]=i
		x_liwc=liwc_feature(lex, cats_to_feature, texts)
		# save pik file
		with open(pikPath, 'wb') as outfile:
			pickle.dump([x_liwc, cats_to_feature], outfile)
		print('features are written to %s file'%pikPath)
		
	return x_liwc, cats_to_feature

def read_file(file_path, if_remove_pos):
	dataF=open(file_path,'r')
	labels, texts = [], []
	for row in dataF:
		row=row.replace('\n','')
		temp=row.split('\t')
		labels.append(int(temp[1]))
		sentence=temp[2]
		if if_remove_pos:
			sentence=removePOS(sentence)
		texts.append(cleanText(sentence))
	return texts, labels

def load_dataset_text(dataset, if_remove_pos):
	pikPath=dataset
	if if_remove_pos:
		pikPath+='-text-pos.pik'
	else:
		pikPath+='-text.pik'
	labels, texts = [], []
	if os.path.exists(pikPath):
		with open(pikPath,'rb') as f:
			[texts, labels]  = pickle.load(f)
			print('Data is loaded from  %s file'%pikPath)
	else:
		dataset=dataset.split('/')[-1]
		if dataset=='hnda':
			file_path='/GW/D5data-11/lwang/fake/propaganda/data2/news/news-a.tsv'
			texts, labels=read_file(file_path, if_remove_pos)
				
		elif dataset=='hnds':
			file_path='/GW/D5data-11/lwang/fake/propaganda/data2/news/news-s.tsv'
			texts, labels=read_file(file_path, if_remove_pos)
			
		elif dataset=='ira':
			file_path='/GW/D5data-11/lwang/fake/propaganda/data2/tweets/tweets.tsv'
			texts, labels=read_file(file_path, if_remove_pos)
		elif dataset=='sp-a':
			file_path='/GW/D5data-11/lwang/fake/propaganda/data2/speech/speech-a.tsv'
			texts, labels=read_file(file_path, if_remove_pos)
					
		elif dataset=='sp-s':
			file_path='/GW/D5data-11/lwang/fake/propaganda/data2/speech/speech-s.tsv'
			texts, labels=read_file(file_path, if_remove_pos)
		# save pik file
		with open(pikPath, 'wb') as outfile:
			pickle.dump([texts, labels], outfile)
		print('features are written to %s file'%pikPath)
		
	return texts, labels

def load_dataset(dataset, fold_num):
	texts, labels, splits=[],[], []
	texts, labels=load_dataset_text(dataset, False)
	labels=np.asarray(labels)
	texts=np.asarray(texts)
	#5 fold training-testing splits
	skf = StratifiedKFold(n_splits=fold_num, random_state=0, shuffle=True)
	for train_index, test_index in skf.split(texts, labels):
		splits.append([texts[train_index], labels[train_index], texts[test_index], labels[test_index]])
	
	return splits
