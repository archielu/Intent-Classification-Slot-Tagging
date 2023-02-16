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
import random
import numpy as np


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def training(args, model, data_loader, optimizer):
	model.train()

	for iteration, batch in enumerate(data_loader):
		optimizer.zero_grad()

		batch['text'] = batch['text'].to(args.device)
		batch['intent'] = batch['intent'].to(args.device)
		output = model(batch)
		loss = output['loss']
		l = loss.item()
		#print("train loss: %.5f, iteration: %d" %(l,iteration))
		output['loss'].backward()
		optimizer.step()



@torch.no_grad()
def evaluation(args, model, data_loader):
	model.eval()

	val_data_num = 0
	correct_num = 0
	for batch in data_loader:
		batch['text'] = batch['text'].to(args.device)
		batch['intent'] = batch['intent'].to(args.device)
		output = model(batch)  

		target = batch['intent'].detach().cpu().tolist()
		predicted = output['pred_label'].detach().cpu().tolist()

		# Get number of equal element in target ans predicted
		correct_num += (np.array(target) == np.array(predicted)).sum()
		batch_sz = len(target)
		val_data_num += batch_sz


	avg_acc = correct_num / val_data_num
	print("avg_acc: {}".format(avg_acc)) 

	return avg_acc  


def main(args):
	with open(args.cache_dir / "vocab.pkl", "rb") as f:
		vocab: Vocab = pickle.load(f)

	intent_idx_path = args.cache_dir / "intent2idx.json"
	intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

	data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
	data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
	datasets: Dict[str, SeqClsDataset] = {
		split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
		for split, split_data in data.items()
	}
	# TODO: crecate DataLoader for train / dev datasets
	train_loader = DataLoader(datasets["train"], batch_size = args.batch_size, shuffle = True, collate_fn = datasets["train"].collate_fn)
	eval_loader = DataLoader(datasets["eval"], batch_size = args.batch_size, shuffle = True, collate_fn = datasets["eval"].collate_fn)

	embeddings = torch.load(args.cache_dir / "embeddings.pt")
	# TODO: init model and move model to target device(cpu / gpu)
	model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes).to(args.device)

	# TODO: init optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.88)


	epoch_pbar = trange(args.num_epoch, desc="Epoch")
	best_avg_acc = 0
	for epoch in epoch_pbar:
		# TODO: Training loop - iterate over train dataloader and update model weights
		training(args, model, train_loader, optimizer)
		scheduler.step()
		# TODO: Evaluation loop - calculate accuracy and save model weights
		val_avg_acc = evaluation(args, model, eval_loader)
		if val_avg_acc > best_avg_acc:
			# Save  model to .pt file
			torch.save(model.state_dict(), "best-intent.pth")
			print('Save model {}'.format(epoch))
			best_avg_acc = val_avg_acc        
	print("best accuracy: {}".format(best_avg_acc))
	# TODO: Inference on test set


def parse_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument(
		"--data_dir",
		type=Path,
		help="Directory to the dataset.",
		default="./data/intent/",
	)
	parser.add_argument(
		"--cache_dir",
		type=Path,
		help="Directory to the preprocessed caches.",
		default="./cache/intent/",
	)
	parser.add_argument(
		"--ckpt_dir",
		type=Path,
		help="Directory to save the model file.",
		default="./ckpt/intent",
	)

	# data
	parser.add_argument("--max_len", type=int, default=128)

	# model
	parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--num_layers", type=int, default=2)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--bidirectional", type=bool, default=True)

	# optimizer
	parser.add_argument("--lr", type=float, default=1e-3)

	# data loader
	parser.add_argument("--batch_size", type=int, default=128)

	# training
	parser.add_argument(
		"--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
	)
	parser.add_argument("--num_epoch", type=int, default=300)
	parser.add_argument("--step_size", type=int, default=80)
	parser.add_argument("--random_seed", type=int, default=1)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	random.seed(args.random_seed)
	args.ckpt_dir.mkdir(parents=True, exist_ok=True)
	main(args)

# Reference: https://github.com/ntu-adl-ta/ADL21-HW1
