import argparse
import json
import numpy as np
import os
import cPickle as pickle
from util.BatchLoaderUnk import BatchLoaderUnk, Tokens
from model.LSTMCNN import LSTMCNN, load_model
from math import exp

Train, Validation, Test = 0, 1, 2

def main(opt):
    loader = BatchLoaderUnk(opt.tokens, opt.data_dir, opt.batch_size, opt.seq_length, opt.max_word_l, opt.n_words, opt.n_chars)
    opt.word_vocab_size = min(opt.n_words, len(loader.idx2word))
    opt.char_vocab_size = min(opt.n_chars, len(loader.idx2char))
    opt.max_word_l = loader.max_word_l
    print 'Word vocab size: ', opt.word_vocab_size, \
        ', Char vocab size: ', opt.char_vocab_size, \
        ', Max word length (incl. padding): ', opt.max_word_l

    # define the model
    if not opt.skip_train:
        print 'creating an LSTM-CNN with ', opt.num_layers, ' layers'
        model = LSTMCNN(opt)
            # make sure output directory exists
        if not os.path.exists(opt.checkpoint_dir):
            os.makedirs(opt.checkpoint_dir)
        pickle.dump(opt, open('{}/{}.pkl'.format(opt.checkpoint_dir, opt.savefile), "wb"))
        model.save('{}/{}.json'.format(opt.checkpoint_dir, opt.savefile))
        model.fit_generator(loader.next_batch(Train), loader.split_sizes[Train]*loader.batch_size, opt.max_epochs,
                            loader.next_batch(Validation), loader.split_sizes[Validation]*loader.batch_size, opt)
        model.save_weights('{}/{}.h5'.format(opt.checkpoint_dir, opt.savefile), overwrite=True)
    else:
        model = load_model('{}/{}.json'.format(opt.checkpoint_dir, opt.savefile))
        model.load_weights('{}/{}.h5'.format(opt.checkpoint_dir, opt.savefile))
        print model.summary()

    # evaluate on full test set.
    test_perp = model.evaluate_generator(loader.next_batch(Test), loader.split_sizes[Test]*loader.batch_size)
    print 'Perplexity on test set: ', exp(test_perp)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a word+character-level language model')
    # data
    parser.add_argument('--data_dir', type=str, default='data/ptb', help='data directory. Should contain train.txt/valid.txt/test.txt with input data')
    # model params
    parser.add_argument('--rnn_size', type=int, default=650, help='size of LSTM internal state')
    parser.add_argument('--use_words', type=int, default=1, help='use words (1=yes)')
    parser.add_argument('--use_chars', type=int, default=0, help='use characters (1=yes)')
    parser.add_argument('--highway_layers', type=int, default=2, help='number of highway layers')
    parser.add_argument('--word_vec_size', type=int, default=650, help='dimensionality of word embeddings')
    parser.add_argument('--char_vec_size', type=int, default=15, help='dimensionality of character embeddings')
    parser.add_argument('--feature_maps', type=int, nargs='+', default=[50,100,150,200,200,200,200], help='number of feature maps in the CNN')
    parser.add_argument('--kernels', type=int, nargs='+', default=[1,2,3,4,5,6,7], help='conv net kernel widths')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the LSTM')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout. 0 = no dropout')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1, help='starting learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--decay_when', type=float, default=1, help='decay if validation perplexity does not improve by more than this much')
    parser.add_argument('--batch_norm', type=int, default=0, help='use batch normalization over input embeddings (1=yes)')
    parser.add_argument('--seq_length', type=int, default=35, help='number of timesteps to unroll for')
    parser.add_argument('--batch_size', type=int, default=20, help='number of sequences to train on in parallel')
    parser.add_argument('--max_epochs', type=int, default=25, help='number of full passes through the training data')
    parser.add_argument('--max_grad_norm', type=float, default=5, help='normalize gradients at')
    parser.add_argument('--max_word_l', type=int, default=65, help='maximum word length')
    parser.add_argument('--n_words', type=int, default=30000, help='max number of words in model')
    parser.add_argument('--n_chars', type=int, default=100, help='max number of char in model')
    # bookkeeping
    parser.add_argument('--seed', type=int, default=3435, help='manual random number generator seed')
    parser.add_argument('--print_every', type=int, default=500, help='how many steps/minibatches between printing out the loss')
    parser.add_argument('--save_every', type=int, default=5, help='save every n epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='cv', help='output directory where checkpoints get written')
    parser.add_argument('--savefile', type=str, default='keras1.0_nozero', help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
    parser.add_argument('--EOS', type=str, default='+', help='<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
    parser.add_argument('--skip_train', default=False, help='skip training', action='store_true')

    # parse input params
    params = parser.parse_args()
    np.random.seed(params.seed)

    assert params.use_words == 1 or params.use_words == 0, '-use_words has to be 0 or 1'
    assert params.use_chars == 1 or params.use_chars == 0, '-use_chars has to be 0 or 1'
    assert (params.use_chars + params.use_words) > 0, 'has to use at least one of words or chars'

    # global constants for certain tokens
    params.tokens = Tokens(
        EOS=params.EOS,
        UNK='|',    # unk word token
        START='{',  # start-of-word token
        END='}',    # end-of-word token
        ZEROPAD=' ' # zero-pad token
    )

    print 'parsed parameters:'
    print json.dumps(vars(params), indent = 2)

    main(params)
