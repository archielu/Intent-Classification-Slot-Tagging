import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagging
from utils import Vocab

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

from seqeval.scheme import IOB2
import random




def main(args):
	with open(args.cache_dir / "vocab.pkl", "rb") as f:
		vocab: Vocab = pickle.load(f)

	tag_idx_path = args.cache_dir / "tag2idx.json"
	tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
	
	data = json.loads(args.test_file.read_text())
	dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
	# TODO: crecate DataLoader for test dataset
	test_loader = DataLoader(dataset, args.batch_size, shuffle = False, collate_fn = dataset.collate_fn)
	embeddings = torch.load(args.cache_dir / "embeddings.pt")
	model = SeqTagging(embeddings,args.hidden_size,args.num_layers,args.dropout,args.bidirectional,dataset.num_classes,)
	model.eval()

	model.load_state_dict(torch.load("./best-slot.pth"))
	model = model.to(args.device)
	p = []
	t = []
	for batch in test_loader:
		batch['tokens'] = batch['tokens'].to(args.device)
		batch['tags'] = batch['tags'].to(args.device)
		output_dict = model(batch)  


		target = batch['tags'].detach().cpu()
		predicted = output_dict['pred_label'].detach().cpu()
		batch_sz = batch['len'].size()[0]
		for i in range(batch_sz):
			pred = predicted[i][:batch['len'][i].item()].tolist()
			tar = target[i][:batch['len'][i].item()].tolist()
			pred = [dataset.idx2label(e) for e in pred]
			tar = [dataset.idx2label(e) for e in tar]
			p.append(pred)
			t.append(tar)

	print(classification_report(t, p, mode='strict', scheme=IOB2))





def parse_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--test_file",type=Path,help="Path to the test file.",required=True)
	parser.add_argument("--data_dir",type=Path,help="Directory to the dataset.",default="./data/slot/")
	parser.add_argument("--cache_dir",type=Path,help="Directory to the preprocessed caches.",default="./cache/slot/")
	parser.add_argument("--ckpt_dir",type=Path,help="Directory to save the model file.",default="./best-slot.pth")
	#parser.add_argument("--pred_file", type=Path, default="./predict/slot/")

	# data
	parser.add_argument("--max_len", type=int, default=128)

	# model
	parser.add_argument("--hidden_size", type=int, default=512)
	parser.add_argument("--num_layers", type=int, default=2)
	parser.add_argument("--dropout", type=float, default=0.15)
	parser.add_argument("--bidirectional", type=bool, default=True)

	# data loader
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu")
	parser.add_argument("--num_epoch", type=int, default=300)
	parser.add_argument("--step_size", type=int, default=50)
	parser.add_argument("--random_seed", type=int, default=45)
	parser.add_argument("--lr", type=float, default=1e-3)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	random.seed(args.random_seed)
	main(args)



# Reference: https://github.com/ntu-adl-ta/ADL21-HW1


