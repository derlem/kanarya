import theano.tensor as T
import theano
import sys
import numpy as np
import layer
import optimize


class BaseModel(object):
    def __init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='LINEAR', dropout_rate=0.0):
        self.n_in = int(n_in)
        self.n_out = int(n_out)

        self.n_layers = len(hidden_layer_size)

        self.dropout_rate = dropout_rate
        self.is_train = T.iscalar('is_train')

        assert len(hidden_layer_size) == len(hidden_layer_type)

        self.x = T.matrix('x')
        self.y = T.matrix('y')

        self.layers = []
        self.params = []

        rng = np.random.RandomState(123)

        for i in range(self.n_layers):
            layer_name = hidden_layer_type[i] + '_' + str(i+1)
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_size[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.layers[i - 1].output

                if hidden_layer_type[i - 1] == 'BLSTM':
                    input_size = hidden_layer_size[i - 1] * 2

            if hidden_layer_type[i] == 'GRU':
                hidden_layer = layer.GRU(rng, layer_input, input_size, hidden_layer_size[i],
                                         p=self.dropout_rate, training=self.is_train, name=layer_name)
            elif hidden_layer_type[i] == 'LSTM':
                hidden_layer = layer.LSTM(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                          training=self.is_train, name=layer_name)
            elif hidden_layer_type[i] == 'BLSTM':
                hidden_layer = layer.BiLSTM(rng, layer_input, input_size, hidden_layer_size[i], hidden_layer_size[i],
                                            p=self.dropout_rate, training=self.is_train, name=layer_name)
            elif hidden_layer_type[i] == 'TANH':
                hidden_layer = layer.FFLayer(rng, layer_input, input_size, hidden_layer_size[i], activation=T.tanh,
                                             p=self.dropout_rate, training=self.is_train, name=layer_name)
            elif hidden_layer_type[i] == 'SIGMOID':
                hidden_layer = layer.FFLayer(rng, layer_input, input_size, hidden_layer_size[i],
                                             activation=T.nnet.sigmoid,
                                             p=self.dropout_rate, training=self.is_train, name=layer_name)
            else:
                print(
                    "This hidden layer type: %s is not supported right now! \n "
                    "Please use one of the following: LSTM, GRU, TANH, SIGMOID\n" % (
                        hidden_layer_type[i]))
                sys.exit(1)

            self.layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

        input_size = hidden_layer_size[-1]

        if hidden_layer_type[-1] == 'BLSTM':
            input_size = hidden_layer_size[-1] * 2

        layer_name = 'Output_Layer'
        if output_type == 'LINEAR':
            self.final_layer = layer.FFLayer(rng, self.layers[-1].output, input_size, self.n_out,
                                             activation=None, name=layer_name)
        elif output_type == 'SIGMOID':
            self.final_layer = layer.FFLayer(rng, self.layers[-1].output, input_size, self.n_out,
                                             activation=T.nnet.sigmoid, name=layer_name)
        else:
            print(
                "This output layer type: %s is not supported right now! \n Please use one of the following: LINEAR\n" % (
                    output_type))
            sys.exit(1)

        self.params.extend(self.final_layer.params)

        # TODO check if you can move self.updates to somewhere else
        self.updates = {}
        for param in self.params:
            self.updates[param] = theano.shared(value=np.zeros(param.get_value(borrow=True).shape,
                                                               dtype=T.config.floatX), name='updates')  # @UndefinedVariable

        # self.cost = T.mean(T.sum((self.final_layer.output - self.y) ** 2, axis=1))
        self.cost = T.mean(T.nnet.binary_crossentropy(self.final_layer.output, self.y))

    def build_steps(self, train_shared_xy, valid_shared_xy, opt='adam', exc_params=None):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        lr = T.scalar('lr', dtype=T.config.floatX)  # @UndefinedVariable
        mom = T.scalar('mom', dtype=T.config.floatX)  # momentum @UndefinedVariable

        if exc_params is not None:
            self.params = [p for p in self.params for x in exc_params if x in p.name]

        gparams = T.grad(self.cost, self.params)

        if opt == 'sgd':
            updates = optimize.sgd_opt(self.updates, self.params, gparams, lr, mom)
        elif opt == 'adam':
            updates = optimize.adam_opt(self.params, gparams, lr, mom)
        else:
            print(
                "This optimization method: %s is not supported right now! \n " % opt)
            sys.exit(1)

        train_model = theano.function(inputs=[lr, mom],
                                      outputs=[self.cost],
                                      updates=updates,
                                      givens={self.x: train_set_x,
                                              self.y: train_set_y,
                                              self.is_train: np.cast['int32'](1)},
                                      on_unused_input='ignore')
        self.diff = T.sum(T.abs_((self.final_layer.output - self.y)))
        valid_model = theano.function(inputs=[],
                                      outputs=[self.cost, self.diff],
                                      givens={self.x: valid_set_x,
                                              self.y: valid_set_y,
                                              self.is_train: np.cast['int32'](0)},
                                      on_unused_input='ignore')

        return train_model, valid_model

    def parameter_prediction(self, test_set_x):

        # dnn_model returned by the cPickle.load() makes self.x float64 so we convert our test files to float64
        # test_set_x = test_set_x.astype(np.float64)

        test_out = theano.function([], self.final_layer.output,
                                   givens={self.x: test_set_x, self.is_train: np.cast['int32'](0)},
                                   on_unused_input='ignore')

        predict_parameter = test_out()

        return predict_parameter
