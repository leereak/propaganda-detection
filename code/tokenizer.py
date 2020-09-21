import re

class Tokenizer:
	def __init__(self):
		self.trimmed = False
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0:'PAD'} #padding with 'PAD'
		self.num_words = 1  # 1 for PAD 
	
	def fit_on_texts(self, all_text_list):
		#clean_text=self.normalize_string(all_text)
		#clean_text=all_text
		for all_text in all_text_list:
			for word in all_text.split(' '):
				self.addWord(word)
		print('The vocabulary size is %s '%self.num_words)
	
	# Lowercase and remove non-letter characters
	#def normalize_string(self, s):
		#s = s.lower()
		#s = re.sub(r"([.!?])", r" \1", s)
		#s = re.sub(r"[^a-zA-Z]+", r" ", s)
		#return s
	
	# add a word to the dict
	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2count[word] += 1

	# Remove words below a certain count threshold
	def trim(self, min_count):
		if self.trimmed:
			return
		self.trimmed = True
		keep_words = []
		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))
		# Reinitialize dictionaries
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0:'PAD'}
		self.num_words = 1
		for word in keep_words:
			self.addWord(word)
	
	#transform the text to id sequences with padding
	def texts_to_sequences(self, texts, padding=128):
		if padding<2:
			print('padding is too small')
			return
		seqs=[0 for i in range(padding)]
		#clean_text=self.normalize_string(texts)
		i=0
		for word in texts.split(' '):
			if i>=padding:
				break
			if word in self.word2index:
				word_id=self.word2index[word]
				seqs[i]=word_id
				i+=1
		return seqs
