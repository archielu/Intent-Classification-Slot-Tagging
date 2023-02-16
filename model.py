from typing import Dict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
	def __init__(
		self,
		embeddings: torch.tensor,
		hidden_size: int,
		num_layers: int,
		dropout: float,
		bidirectional: bool,
		num_class: int,
	) -> None:
		super(SeqClassifier, self).__init__()
		self.classifier_input_size = (2*hidden_size if bidirectional else hidden_size)
		self.embed = Embedding.from_pretrained(embeddings, freeze=False)
		self.embed_dim = embeddings.size()[1]
		self.bidirectional = bidirectional

		# TODO: model architecture
		self.lstm = nn.LSTM(input_size = self.embed_dim, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout, bidirectional = bidirectional)
		self.dp= nn.Dropout(dropout)
		self.classifier  = nn.Linear(self.classifier_input_size, num_class)
		self.sftmx = nn.Softmax(dim = 1)

	def forward(self, batch) -> Dict[str, torch.Tensor]:
		'''
		inputs: [batch_size,max_len]
		target:	[batch_size]
		feature [num_layers * 2,batch_size,hidden_size] -> [batch_size, hidden_size * 2]
		prob: [batch_size, num_classes]
		'''
		# TODO: implement model forward
		inputs = batch['text']
		inputs = self.embed(inputs)
		target = batch['intent']
		# Pack inputs 
		packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, batch['len'], batch_first=True, enforce_sorted=False)

		# model
		outputs, (hidden_state, _) = self.lstm(packed_inputs)
		feature = hidden_state

		# hidden state size: (D*num_layers,batch_size,hidden_size)

		# By default, self.bidirection is True
		feature = hidden_state[-1]
		if self.bidirectional == True:
			feature = torch.cat((hidden_state[-1],hidden_state[-2]), axis = -1)

		feature = self.dp(feature)
		prob = self.classifier(feature)
		# Get better performance if removing softmax
		#prob = self.sftmx(feature)

		# Get predicted labels and caculate loss
		output = {}
		output['loss'] =  F.cross_entropy(prob,target)
		output['pred_label'] = torch.max(prob,1)[1]
		 
		return output             


class SeqTagging(torch.nn.Module):
	def __init__(
		self,
		embeddings: torch.tensor,
		hidden_size: int,
		num_layers: int,
		dropout: float,
		bidirectional: bool,
		num_class: int,
	) -> None:
		super(SeqTagging, self).__init__()
		self.classifier_input_size = (2*hidden_size if bidirectional else hidden_size)
		self.embed = Embedding.from_pretrained(embeddings, freeze=False)
		self.embed_dim = embeddings.size()[1]
		

		# TODO: model architecture
		self.lstm = nn.LSTM(input_size = self.embed_dim, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout, bidirectional = bidirectional)
		self.dp= nn.Dropout(dropout)
		self.classifier  = nn.Linear(self.classifier_input_size, num_class)
		self.sftmx = nn.Softmax(dim = 1)


	def forward(self, batch) -> Dict[str, torch.Tensor]:
		# TODO: implement model forward
		# target: tensor batch_size * max_len
		inputs = batch['tokens']
		inputs = self.embed(inputs)
		target = batch['tags']
		# Pack inputs
		packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, batch['len'], batch_first=True, enforce_sorted=False)
		outputs, _ = self.lstm(packed_inputs)
		# Unpack outputs : place padding back
		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True) 

		feature = self.dp(outputs)
		prob = self.classifier(feature)
		#get better performance after removing softmax
		#prob = self.sftmx(feature) # Size prob : [batch_size, max_len, num_class]
	
		## Caculate loss
		loss = 0
		batch_sz = batch['len'].size()[0]
		for i in range(batch_sz):
			# i is batch no.
			sentence_len = batch['len'][i].item()
			logits = prob[i][:sentence_len]
			tars = target[i][:sentence_len]
			loss += F.cross_entropy(logits, tars)

		output = {}
		output['loss'] = loss
		output['pred_label'] = torch.max(prob,-1)[1]

		return output  

# Size outputs : [batch_size, max_len, D*hidden_size]				   

