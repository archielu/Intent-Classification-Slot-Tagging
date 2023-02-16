import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn

import torch
from torch.nn import Embedding
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from model import SeqClassifier
import csv
import random
import os



def main(args):
	with open(args.cache_dir / "vocab.pkl", "rb") as f:
		vocab: Vocab = pickle.load(f)

	intent_idx_path = args.cache_dir / "intent2idx.json"
	intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
	embeddings = torch.load(args.cache_dir / "embeddings.pt")

	data = json.loads(args.test_file.read_text())
	dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
	

	model = SeqClassifier(embeddings,args.hidden_size,args.num_layers,args.dropout,args.bidirectional,dataset.num_classes)
	model.eval()

	# load weights into model
	model.load_state_dict(torch.load("best-intent.pth"))
	model = model.to(args.device)

	# TODO: crecate DataLoader for test dataset
	data_loader = DataLoader(dataset, args.batch_size, shuffle = False, collate_fn = dataset.collate_fn)
	os.makedirs(os.path.dirname(args.pred_file), exist_ok=True)
	# TODO: predict dataset

	fields = ['id','intent']
	with open(args.pred_file,'w') as f:
		write = csv.writer(f)
		write.writerow(fields)
		for batch in data_loader:
			batch['text'] = batch['text'].to(args.device)
			batch['intent'] = batch['intent'].to(args.device)
			output = model(batch)  
			predicted = output['pred_label'].detach().cpu().tolist()

			for idx in range(len(batch['id'])):
				line = []
				line.append(batch['id'][idx])
				line.append(dataset.idx2label(predicted[idx]))
				write.writerow(line)





def parse_args() -> Namespace:
	parser = ArgumentParser()

	parser.add_argument("--cache_dir",type=Path,help="Directory to the preprocessed caches.",default="./cache/intent/")


	# data
	parser.add_argument("--max_len", type=int, default=128)

	parser.add_argument("--ckpt_path",type=Path,help="Path to model checkpoint.",default = "./ckpt/intent/")

	# Hyperparameter
	parser.add_argument("--random_seed", type=int, default=1)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--num_epoch", type=int, default=300)
	parser.add_argument("--step_size", type=int, default=80)

	# model
	parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--num_layers", type=int, default=2)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--bidirectional", type=bool, default=True)



	# data loader
	parser.add_argument("--batch_size", type=int, default=128)

	parser.add_argument(
		"--test_file",
		type=Path,
		help="Path to the test file.",
		required=True
	)


	parser.add_argument(
		"--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
	)

	parser.add_argument("--pred_file",required=True)
	
	
	
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	random.seed(args.random_seed)
	main(args)


# Reference: https://github.com/ntu-adl-ta/ADL21-HW1