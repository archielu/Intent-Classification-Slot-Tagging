from typing import List, Dict
from torch.utils.data import Dataset
import torch
from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
	def __init__(
		self,
		data: List[Dict],
		vocab: Vocab,
		label_mapping: Dict[str, int],
		max_len: int,
	):
		self.data = data
		self.vocab = vocab
		self.label_mapping = label_mapping
		self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
		self.max_len = max_len

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, index) -> Dict:
		instance = self.data[index]
		return instance

	@property
	def num_classes(self) -> int:
		return len(self.label_mapping)

	def collate_fn(self, samples: List[Dict]) -> Dict:    
		# TODO: implement collate_fn
		'''
		samples: List[Dict] Dict Key: id,text,intent
		convert samples to batch(Dict), Key: id,text,intent,len
		'''
		batch = {}
		batch['id'] = []
		batch['text'] = []
		batch['intent'] = []
		batch['len'] = []
		for s in samples:
			batch['id'].append(s['id'])
			batch['text'].append(s['text'].split())
		for s in batch['text']:
			batch['len'].append(min(len(s),self.max_len))
		batch['len'] = torch.tensor(batch['len'])

		if 'intent' not in samples[0].keys():   #test
			batch['intent'] = torch.tensor([0] * len(samples))
		else:   #train & eval
			for s in samples:
				batch['intent'].append(self.label_mapping[s['intent']])
			batch['intent'] = torch.tensor(batch['intent'])
		   
		batch['text'] = torch.tensor(self.vocab.encode_batch(batch['text'], self.max_len))

		return batch


		#raise NotImplementedError

	def label2idx(self, label: str):
		return self.label_mapping[label]

	def idx2label(self, idx: int):
		return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
	ignore_idx = -100

	def collate_fn(self, samples):
		# TODO: implement collate_fn
		'''
		samples: List[Dict] Dict Key: id,tokens,tags
		convert samples to batch(Dict), Key: id,tokens,tags,len
		'''
		batch = {}
		batch['id']=[]
		batch['tokens'] = []
		batch['tags'] = []
		batch['len'] = []
		for s in samples:
			batch['id'].append(s['id'])
			batch['tokens'].append(s['tokens'])
		for s in batch['tokens']:
			batch['len'].append(min(len(s),self.max_len))
		batch['len'] = torch.tensor(batch['len'])
		
		if 'tags' not in samples[0].keys(): # test
			batch['tags'] = torch.tensor([[0] * self.max_len] * len(samples))
		else: # train & eval
			for s in samples:
				batch['tags'].append(s['tags'])
			batch['tags'] = [[self.label_mapping[tag] for tag in s] for s in batch['tags']]
			batch['tags'] = torch.tensor(pad_to_len(batch['tags'], self.max_len, -1))
	
		batch['tokens'] = torch.tensor(self.vocab.encode_batch(batch['tokens'], self.max_len))

		return batch

		#raise NotImplementedError

# Reference: https://github.com/ntu-adl-ta/ADL21-HW1
