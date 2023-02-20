
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

# Q1 : Data Processing
## Tokenize
Use sample code, load text from train/eval .json, splt text to get word, choose top 10000 words by frequency as vocab. Assign id for elements in vocab. ID : 0 : **[PAD]**, ID : 1 : **[UNK]**, for words not in vocab. Function **label2idx** converts word to idx.

## Word Embedding
Use Glove, mapping words to 300 dimension tensor according to glove840B.300d.txt. If word is not in glove, assign a random tensor to it.


---


# Q2 : Intent Classification Model
## Model
### LSTM
將packed inputs, $\ h_{in}$, $\ c_{in}$ 送進 nn.LSTM，並得到outputs o,(h,c)。 忽略 o 跟 c ， h(hidden state) 為我們要的feature。 因為設定是 bi-LSTM，將最後一層layer雙向的 h concatenate:
```
feature = concat(h[-1],h[-2])
```
hidden_state 每一層的 size 為 hidden_size ， 所以 feature 最終的 size 為 2*hidden_size 。
### Dropout
過一層dropout層，將rate設成0.2。代表feature中每一個neural有0.2的機率會被歸零。

### Linear
nn.Linear 當作 classifier，輸入為feature，輸出維度大小為num_intent_classes。
### Softmax
本來在classifier出來的logits會過softmax層，但是後來發現拿掉的效果比較好。猜測是因為pytorch 的 F.cross_entropy 是由 softmax 跟  NLL loss 組成，所以不用另外再過softmax。 
### Hyperparameter
**hidden_size** : 256
**num_layers** : 2
**batch_siz**e: 128
**num_epoch** : 300
**lstm_dropout** : 0.2
**random_seed** : 1

### Loss Function
將 classfier 輸出的結果與 target_labels 做cross_entropy
```
loss = cross_entropy(classifier_out, target_labels)
```

### Optimization Algorithm
Adam， lr = 0.001。使用stepLR， step_size = 80， gamma = 0.88。

### Performance
kaggle Public : **0.91644**
kaggle Private : **0.91333**


---


# Q3 : Slot Tagging Model
## Model
### LSTM
將packed inputs, $\ h_{in}$, $\ c_{in}$ 送進 nn.LSTM，並得到outputs o,(h,c)。 與Q2不同，這次每個token都需要提取feature，所忽略 h 跟 c ， o 為我們要的features。 因為前面有將 data packed，所以 o 在送進之後的 layer 前需要先 unpack 回我們熟悉的結構。

### Dropout
過一層dropout層，將rate設成0.15。代表feature中每一個neural有0.15的機率會被歸零。

### Linear
nn.Linear 當作 classifier，輸入為feature，輸出維度大小為num_tag_classes。

### Hyperparameter
**hidden_size** : 512
**num_layers** : 2
**batch_siz**e: 128
**num_epoch** : 300
**lstm_dropout** : 0.15
**random_seed** : 45

### Loss Function
在batch 中的每個 sequence 中的每的 token 都會有自己的 feature，在經過classifier之後，為每個 token 與真實 target計算cross_entropy。 最後，將 batch 中所有 seq 的所有token對應的 cross_entropy相加形成 loss。

### Optimization Algorithm
Adam， lr = 0.001。使用stepLR， step_size = 50， gamma = 0.88。

### Performance
kaggle Public : **0.78123**
kaggle Private : **0.77974**


---




# Q4 : Sequence Tagging Evaluation
![](https://i.imgur.com/MLkzQNz.png)



**joint accuracy** =  $\frac{number \ of \ correct \ sentences(all \  tokens \ in \ the \ sentence \ predict \ correctly)}{number \ of \ all \ sentences}$
**token accuracy** = $\frac{number \ of \ correct \ tokens}{number \ of \ all \ tokens}$

Seqeval 將結果轉換成(tag,begin,end)形式，接著計算數據：



| Metric | Definition | Formula |
| -------- | -------- | -------- |
| Precision    | $\frac{TP}{TP+FP}$   |  $\frac{預測tag 且確實為該tag的個數}{預測為tag的數量}$    |
| Recall   |    $\frac{TP}{TP+FN}$  |  $\frac{預測tag且確實為該tag的個數}{真實為tag的數量}$   |
| f1-score    | $\frac{2}{\frac{1}{precision} + \frac{1}{recall}}$     |  -   |
| micro avg    |  -    |   $\frac{所有tag的TP}{所有tag(TP+FP)}、 \frac{所有tag的 TP}{所有tag(TP+FN)}$  |
| macro avg   |  -   | 對所有 recall、precision 平均   |
| weighted avg     |  -    | 將數據依tag數量做weighted sum     |
| support    |   -   | 真實tag為該tag的數量    |



# Q5
在training的過程中，固定 train 300 個 epoch，這很可能造成 over-fitting。所以在選model的時候會選的是在eval得到最高的accuracy 的 epoch model 。
