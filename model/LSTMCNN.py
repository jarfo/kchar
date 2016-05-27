from math import exp

from keras.models import Model, model_from_json
from keras.layers import Input, Embedding, TimeDistributed, Dense, Dropout, Reshape, Merge, Highway, LSTM, Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras import backend as K

class sSGD(SGD):
    def __init__(self, scale=1., **kwargs):
        super(sSGD, self).__init__(**kwargs)
        self.scale = scale;
    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if self.scale != 1.:
            grads = [g*K.variable(self.scale) for g in grads]
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [K.switch(norm >= self.clipnorm, g * self.clipnorm / norm, g) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads
    
class sModel(Model):
    def fit_generator(self, generator, samples_per_epoch, nb_epoch, validation_generator, nb_val_samples, opt):
        val_losses = []
        lr = K.get_value(self.optimizer.lr)
        for epoch in range(nb_epoch):
            super(sModel, self).fit_generator(generator, samples_per_epoch, 1, verbose=1)
            val_loss = exp(self.evaluate_generator(validation_generator, nb_val_samples))
            val_losses.append(val_loss)
            print 'Epoch {}/{}. Validation loss: {}'.format(epoch + 1, nb_epoch, val_loss)
            if len(val_losses) > 2 and (val_losses[-2] - val_losses[-1]) < opt.decay_when:
                lr *= opt.learning_rate_decay
                K.set_value(self.optimizer.lr, lr)
            if epoch == nb_epoch-1 or epoch % opt.save_every == 0:
                savefile = '%s/lm_%s_epoch%d_%.2f.h5' % (opt.checkpoint_dir, opt.savefile, epoch + 1, val_loss)
                self.save_weights(savefile)
    @property
    def state_updates_value(self):
        return [K.get_value(a[0]) for a in self.state_updates]

    def set_states_value(self, states):
        return [K.set_value(a[0], state) for a, state in zip(self.state_updates, states)]

    def save(self, name):
        json_string = self.to_json()
        with open(name, 'wt') as f:
            f.write(json_string)

def load_model(name):
    with open(name, 'rt') as f:
        json_string = f.read()
    model = model_from_json(json_string, custom_objects={'sModel': sModel})
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD)
    model.compile(loss='categorical_crossentropy', optimizer=SGD)
    return model


def CNN(seq_length, length, input_size, feature_maps, kernels, x):
    
    concat_input = []
    for feature_map, kernel in zip(feature_maps, kernels):
        reduced_l = length - kernel + 1
        conv = Convolution2D(feature_map, 1, kernel, activation='tanh', dim_ordering='tf')(x)
        maxp = MaxPooling2D((1, reduced_l), dim_ordering='tf')(conv)
        concat_input.append(maxp)

    x = Merge(mode='concat')(concat_input)
    x = Reshape((seq_length, sum(feature_maps)))(x)
    return x

def LSTMCNN(opt):
    # opt.seq_length = number of time steps (words) in each batch
    # opt.rnn_size = dimensionality of hidden layers
    # opt.num_layers = number of layers
    # opt.dropout = dropout probability
    # opt.word_vocab_size = num words in the vocab
    # opt.word_vec_size = dimensionality of word embeddings
    # opt.char_vocab_size = num chars in the character vocab
    # opt.char_vec_size = dimensionality of char embeddings
    # opt.feature_maps = table of feature map sizes for each kernel width
    # opt.kernels = table of kernel widths
    # opt.length = max length of a word
    # opt.use_words = 1 if use word embeddings, otherwise not
    # opt.use_chars = 1 if use char embeddings, otherwise not
    # opt.highway_layers = number of highway layers to use, if any
    # opt.batch_size = number of sequences in each batch

    if opt.use_words:
        word = Input(batch_shape=(opt.batch_size, opt.seq_length), dtype='int32', name='word')
        word_vecs = Embedding(opt.word_vocab_size, opt.word_vec_size, input_length=opt.seq_length)(word)

    if opt.use_chars:
        chars = Input(batch_shape=(opt.batch_size, opt.seq_length, opt.max_word_l), dtype='int32', name='chars')
        chars_embedding = TimeDistributed(Embedding(opt.char_vocab_size, opt.char_vec_size, name='chars_embedding'))(chars)
        cnn = CNN(opt.seq_length, opt.max_word_l, opt.char_vec_size, opt.feature_maps, opt.kernels, chars_embedding)
        if opt.use_words:
            x = Merge(mode='concat')([cnn, word_vecs])
            inputs = [chars, word]
        else:
            x = cnn
            inputs = chars
    else:
        x = word_vecs
        inputs = word

    if opt.batch_norm:
        x = BatchNormalization()(x)

    for l in range(opt.highway_layers):
        x = TimeDistributed(Highway(activation='relu'))(x)

    for l in range(opt.num_layers):
        x = LSTM(opt.rnn_size, activation='tanh', inner_activation='sigmoid', return_sequences=True, stateful=True)(x)

        if opt.dropout > 0:
            x = Dropout(opt.dropout)(x)

    output = TimeDistributed(Dense(opt.word_vocab_size, activation='softmax'))(x)
    
    model = sModel(input=inputs, output=output)
    print model.summary()

    optimizer = sSGD(lr=opt.learning_rate, clipnorm=opt.max_grad_norm, scale=float(opt.seq_length))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model
