## Character-Aware Neural Language Models. A Keras-based implementation

Implementation of the character-based language model proposed in the paper [Character-Aware Neural Language Models](http://arxiv.org/abs/1508.06615) 
(AAAI 2016) using the [Keras](http://keras.io) neural networks library.

The code is based on the [original LUA source code](https://github.com/yoonkim/lstm-char-cnn) for the Torch library.

### Requirements
Code is written in python 2.7 and requires [Keras](https://github.com/fchollet/keras).
```
### Data
Data should be split into `train.txt`, `valid.txt`, and `test.txt`

Each line of the .txt file should be a sentence. The English Penn 
Treebank (PTB) data (Tomas Mikolov's pre-processed version with vocab size equal to 10K,
widely used by the language modeling community) is given as the default.

### Model
You can reproduce the results of the paper as follows

#### Character-level models
Large character-level model (LSTM-CharCNN-Large in the paper).
This is the default: should get ~82 on valid and ~79 on test. Takes ~3.5 hours with Theano (GPU/CuDNN).
```
python train.py --savefile char-large
```

#### Word-level models
Large word-level model (LSTM-Word-Large in the paper).
This should get ~89 on valid and ~85 on test.
```
python train.py --savefile word-large --highway_layers 0 --use_chars 0 --use_words 1
```

### Evaluation
By default `train.py` will evaluate the model on test data after training using the last epoch's model, and also will be slow due to
the way the data is set up.

Evaluation can be performed via the following script:
```
python evaluate.py --model cv/char-large --vocabulary data/ptb/vocab.npz --init init.npy --text data/ptb/test.txt --calc
```
With the --calc option the state of the network is not reset after each sentence, and the mean value of the initial state is saved in the --init file.
Using this cross-sentence information helps in the case of the provided PTB data but it is not useful for sentences in random order.

For this later case, we can evaluate the perplexity using a precomputed --init file as the initial state of the LSTM networks at the beginning of each sentence
```
python evaluate.py --model cv/char-large --vocabulary data/ptb/vocab.npz --init init.npy --text data/ptb/test.txt
```

### Licence
MIT

