import theano.tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from theano import config
import numpy as np


class FFLayer(object):

    def __init__(self, rng, x, n_in, n_out, W=None, b=None, activation=T.tanh, p=0.0, training=0, name='FF'):
        self.x = x

        if p > 0.0:
            if training == 1:
                srng = RandomStreams(seed=123456)
                self.x = T.switch(srng.binomial(size=x.shape, p=p), x, 0)
            else:
                self.x = (1 - p) * x

        if W is None:
            W_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_in),
                                            size=(n_in, n_out)), dtype=T.config.floatX)
            W = theano.shared(value=W_value,
                              name=name+'_W', borrow=True)
        if b is None:
            b = theano.shared(value=np.zeros((n_out,), dtype=T.config.floatX),
                              name=name+'_b', borrow=True)

        self.W = W
        self.b = b

        self.output = T.dot(self.x, self.W) + self.b
        if activation is not None:
            self.output = activation(self.output)

        self.params = [self.W, self.b]


class LSTM(object):

    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, name='LSTM'):
        self.input = x

        if p > 0.0:
            if training == 1:
                srng = RandomStreams(seed=123456)
                self.input = T.switch(srng.binomial(size=x.shape, p=p), x, 0)
            else:
                self.input = (1 - p) * x

        self.n_in = int(n_in)
        self.n_h = int(n_h)

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h), size=(n_h,)), dtype=config.floatX)

        # Input gate weights
        self.W_xi = theano.shared(value=Wx_value, name=name+'_W_xi')
        self.W_hi = theano.shared(value=Wh_value, name=name+'_W_hi')
        self.w_ci = theano.shared(value=Wc_value, name=name+'_w_ci')

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h), size=(n_h,)), dtype=config.floatX)

        # Forget gate weights
        self.W_xf = theano.shared(value=Wx_value, name=name+'_W_xf')
        self.W_hf = theano.shared(value=Wh_value, name=name+'_W_hf')
        self.w_cf = theano.shared(value=Wc_value, name=name+'_w_cf')

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h), size=(n_h,)), dtype=config.floatX)

        # Output gate weights
        self.W_xo = theano.shared(value=Wx_value, name=name+'_W_xo')
        self.W_ho = theano.shared(value=Wh_value, name=name+'_W_ho')
        self.w_co = theano.shared(value=Wc_value, name=name+'_w_co')

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h), size=(n_h,)), dtype=config.floatX)

        # Cell weights
        self.W_xc = theano.shared(value=Wx_value, name=name+'_W_xc')
        self.W_hc = theano.shared(value=Wh_value, name=name+'_W_hc')

        # bias
        self.b_i = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_b_i')
        self.b_f = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_b_f')
        self.b_o = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_b_o')
        self.b_c = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_b_c')

        ### make a layer

        # initial value of hidden and cell state
        self.h0 = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_h0')
        self.c0 = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_c0')

        self.Wix = T.dot(self.input, self.W_xi)
        self.Wfx = T.dot(self.input, self.W_xf)
        self.Wcx = T.dot(self.input, self.W_xc)
        self.Wox = T.dot(self.input, self.W_xo)

        [self.h, self.c], _ = theano.scan(self.step, sequences=[self.Wix, self.Wfx, self.Wcx, self.Wox],
                                          outputs_info=[self.h0, self.c0])

        self.output = self.h
        self.params = [self.W_xi, self.W_hi, self.w_ci,
                       self.W_xf, self.W_hf, self.w_cf,
                       self.W_xo, self.W_ho, self.w_co,
                       self.W_xc, self.W_hc,
                       self.b_i, self.b_f, self.b_o, self.b_c]

    def step(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1=None):
        h_t, c_t = self.lstm_as_activation_function(Wix, Wfx, Wcx, Wox, h_tm1, c_tm1)
        return h_t, c_t

    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1):
        i_t = T.nnet.sigmoid(Wix + T.dot(h_tm1, self.W_hi) + self.w_ci * c_tm1 + self.b_i)  #
        f_t = T.nnet.sigmoid(Wfx + T.dot(h_tm1, self.W_hf) + self.w_cf * c_tm1 + self.b_f)  #

        c_t = f_t * c_tm1 + i_t * T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + self.b_c)

        o_t = T.nnet.sigmoid(Wox + T.dot(h_tm1, self.W_ho) + self.w_co * c_t + self.b_o)

        h_t = o_t * T.tanh(c_t)

        return h_t, c_t


class GRU(object):

    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, name='GRU'):

        self.n_in = int(n_in)
        self.n_h = int(n_h)

        self.input = x

        if p > 0.0:
            if training == 1:
                srng = RandomStreams(seed=123456)
                self.input = T.switch(srng.binomial(size=x.shape, p=p), x, 0)
            else:
                self.input = (1 - p) * x

        self.W_xz = theano.shared(value=np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_in),
                                                              size=(n_in, n_h)), dtype=config.floatX), name=name+'_W_xz')
        self.W_hz = theano.shared(value=np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h),
                                                              size=(n_h, n_h)), dtype=config.floatX), name=name+'_W_hz')

        self.W_xr = theano.shared(value=np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_in),
                                                              size=(n_in, n_h)), dtype=config.floatX), name=name+'_W_xr')
        self.W_hr = theano.shared(value=np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h),
                                                              size=(n_h, n_h)), dtype=config.floatX), name=name+'_W_hr')

        self.W_xh = theano.shared(value=np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_in),
                                                              size=(n_in, n_h)), dtype=config.floatX), name=name+'_W_xh')
        self.W_hh = theano.shared(value=np.asarray(rng.normal(0.0, 1.0 / np.sqrt(n_h),
                                                              size=(n_h, n_h)), dtype=config.floatX), name=name+'_W_hh')

        self.b_z = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_b_z')

        self.b_r = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_b_r')

        self.b_h = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_b_h')

        self.h0 = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_h0')
        self.c0 = theano.shared(value=np.zeros((n_h,), dtype=config.floatX), name=name+'_c0')

        ## pre-compute these for fast computation
        self.Wzx = T.dot(self.input, self.W_xz)
        self.Wrx = T.dot(self.input, self.W_xr)
        self.Whx = T.dot(self.input, self.W_xh)

        [self.h, self.c], _ = theano.scan(self.gru_as_activation_function,
                                          sequences=[self.Wzx, self.Wrx, self.Whx],
                                          outputs_info=[self.h0, self.c0])  #

        self.output = self.h

        self.params = [self.W_xz, self.W_hz, self.W_xr, self.W_hr, self.W_xh, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        # self.L2_cost = (self.W_xz ** 2).sum() + (self.W_hz ** 2).sum() + (self.W_xr ** 2).sum() + (
        # self.W_hr ** 2).sum() + (self.W_xh ** 2).sum() + (self.W_hh ** 2).sum()

    def gru_as_activation_function(self, Wzx, Wrx, Whx, h_tm1, c_tm1=None):

        z_t = T.nnet.sigmoid(Wzx + T.dot(h_tm1, self.W_hz) + self.b_z)
        r_t = T.nnet.sigmoid(Wrx + T.dot(h_tm1, self.W_hr) + self.b_r)
        can_h_t = T.tanh(Whx + r_t * T.dot(h_tm1, self.W_hh) + self.b_h)

        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        c_t = h_t

        return h_t, c_t


class BiLSTM(LSTM):
    def __init__(self, rng, x, n_in, n_h, n_out, p=0.0, training=0, name='biLSTM'):
        fwd = LSTM(rng, x, n_in, n_h, p, training, name)
        bwd = LSTM(rng, x[::-1], n_in, n_h, p, training, name)

        self.params = fwd.params + bwd.params

        self.output = T.concatenate([fwd.output, bwd.output[::-1]], axis=1)