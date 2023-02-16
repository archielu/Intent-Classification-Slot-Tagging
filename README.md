
# ADL22 
## Environment
```
pip3 install -r requirements.in
```

## Prpocessing 
```
bash preprocess.sh
```

## Download my model
```
bash download.sh
```

## Intent Classification
### Train model
```
python3 train_intent.py --data_dir --cache_dir --ckpt_dir --hidden_size --num_layers --dropout --random_seed --device
```

### Reproduce
```
bash test_intent.sh /path/to/test.json /path/to/pred.csv
```


## Slot Tagging
### Train model
```
python3 train_slot.py --data_dir --cache_dir --ckpt_dir --hidden_size --num_layers --dropout --random_seed --device
```

### Reproduce
```
bash test_slot.sh /path/to/test.json /path/to/pred.csv
```
