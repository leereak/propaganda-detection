import torch
from torch.utils.data import Dataset


class myDataset(Dataset):
	def __init__(self, claims, labels):
		self.size=len(claims)
		self.claims = torch.tensor(claims)
		self.labels=torch.tensor(labels)

	def __getitem__(self, index):
		
		return self.claims[index], self.labels[index]

	def __len__(self):
		return self.size


class myDataset_guided(Dataset):
	def __init__(self, claims, labels, guided_masks):
		self.size=len(claims)
		self.claims = torch.tensor(claims)
		self.labels=torch.tensor(labels)
		self.guided_masks=torch.tensor(guided_masks)

	def __getitem__(self, index):
		
		return self.claims[index], self.labels[index], self.guided_masks[index]

	def __len__(self):
		return self.size

class myDataset_pair(Dataset):
	def __init__(self, pos, neg):
		self.size=len(pos)
		self.pos = torch.tensor(pos)
		self.neg=torch.tensor(neg)

	def __getitem__(self, index):
		
		return self.pos[index], self.neg[index]

	def __len__(self):
		return self.size
	
class myDataset_bert_pair(Dataset):
	def __init__(self, input_pos, input_neg, mask_pos, mask_neg):
		self.size=len(input_pos)
		self.input_pos = torch.tensor(input_pos)
		self.input_neg=torch.tensor(input_neg)
		self.mask_pos = torch.tensor(mask_pos)
		self.mask_neg =torch.tensor(mask_neg)

	def __getitem__(self, index):
		
		return self.input_pos[index], self.input_neg[index], self.mask_pos[index], self.mask_neg[index]

	def __len__(self):
		return self.size

class myDataset_bert(Dataset):
	def __init__(self, inputs, masks, labels):
		self.size=len(inputs)
		self.inputs = torch.tensor(inputs)
		self.masks=torch.tensor(masks)
		self.labels=torch.tensor(labels)

	def __getitem__(self, index):
		
		return self.inputs[index], self.masks[index], self.labels[index]

	def __len__(self):
		return self.size