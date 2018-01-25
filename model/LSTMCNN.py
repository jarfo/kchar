from math import exp

from keras import initializers, regularizers, activations, constraints
from keras.engine import Layer, InputSpec
from keras.models import Model, model_from_json
from keras.layers import Input, Embedding, TimeDistributed, Dense, Dropout, Reshape, Concatenate, LSTM, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras import backend as K

class Highway(Layer):
    """Densely connected highway network.
    Highway layers are a natural extension of LSTMs to feedforward networks.
    # Arguments
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Highway Networks](http://arxiv.org/abs/1505.00387v2)
    """

    def __init__(self,
                 init='glorot_uniform',
                 activation=None,
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):

        self.init = initializers.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    shape=(None, input_dim))

        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.W_carry = self.add_weight((input_dim, input_dim),
                                       initializer=self.init,
                                       name='W_carry')
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zero',
                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b_carry = self.add_weight((input_dim,),
                                           initializer='one',
                                           name='b_carry')
        else:
            self.b_carry = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x):
        y = K.dot(x, self.W_carry)
        if self.bias:
            y += self.b_carry
        transform_weight = activations.sigmoid(y)
        y = K.dot(x, self.W)
        if self.bias:
            y += self.b
        act = self.activation(y)
        act *= transform_weight
        output = act + (1 - transform_weight) * x
        return output

    def get_config(self):
        config = {'init': initializers.serialize(self.init),
                  'activation': activations.serialize(self.activation),
                  'W_regularizer': regularizers.serialize(self.W_regularizer),
                  'b_regularizer': regularizers.serialize(self.b_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'W_constraint': constraints.serialize(self.W_constraint),
                  'b_constraint': constraints.serialize(self.b_constraint),
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
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
    def fit_generator(self, generator, steps_per_epoch, epochs, validation_data, validation_steps, opt):
        val_losses = []
        lr = K.get_value(self.optimizer.lr)
        for epoch in range(epochs):
            super(sModel, self).fit_generator(generator, steps_per_epoch, epochs=epoch+1, verbose=1, initial_epoch=epoch)
            val_loss = exp(self.evaluate_generator(validation_data, validation_steps))
            val_losses.append(val_loss)
            print 'Epoch {}/{}. Validation loss: {}'.format(epoch + 1, epochs, val_loss)
            if len(val_losses) > 2 and (val_losses[-2] - val_losses[-1]) < opt.decay_when:
                lr *= opt.learning_rate_decay
                K.set_value(self.optimizer.lr, lr)
            if epoch == epochs-1 or epoch % opt.save_every == 0:
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
    model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD)
    return model


def CNN(seq_length, length, input_size, feature_maps, kernels, x):
    
    concat_input = []
    for feature_map, kernel in zip(feature_maps, kernels):
        reduced_l = length - kernel + 1
        conv = Conv2D(feature_map, (1, kernel), activation='tanh', data_format="channels_last")(x)
        maxp = MaxPooling2D((1, reduced_l), data_format="channels_last")(conv)
        concat_input.append(maxp)

    x = Concatenate()(concat_input)
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
        chars_embedding = Embedding(opt.char_vocab_size, opt.char_vec_size, name='chars_embedding')(chars)
        cnn = CNN(opt.seq_length, opt.max_word_l, opt.char_vec_size, opt.feature_maps, opt.kernels, chars_embedding)
        if opt.use_words:
            x = Concatenate()([cnn, word_vecs])
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
        x = LSTM(opt.rnn_size, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, stateful=True)(x)

        if opt.dropout > 0:
            x = Dropout(opt.dropout)(x)

    output = TimeDistributed(Dense(opt.word_vocab_size, activation='softmax'))(x)

    model = sModel(inputs=inputs, outputs=output)
    model.summary()

    optimizer = sSGD(lr=opt.learning_rate, clipnorm=opt.max_grad_norm, scale=float(opt.seq_length))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

    return model
