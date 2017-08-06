import theano
import collections
import theano.tensor as T
import numpy as np
from theano import config


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def adam_opt(params, gparams, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    # zip just concatenate two lists
    updates = collections.OrderedDict()  # @UndefinedVariable
    i = theano.shared(numpy_floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1) ** i_t
    fix2 = 1. - (1. - b2) ** i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for param, gparam in zip(params, gparams):
        m = theano.shared(param.get_value() * 0.)
        v = theano.shared(param.get_value() * 0.)
        m_t = (b1 * gparam) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(gparam)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = param - (lr_t * g_t)
        updates[m] = m_t
        updates[v] = v_t
        updates[param] = p_t
    updates[i] = i_t
    return updates


def sgd_opt(weight_updates, params, gparams, lr=0.0002, mom=0.9):
    # zip just concatenate two lists
    updates = collections.OrderedDict()  # @UndefinedVariable
    for param, gparam in zip(params, gparams):
        weight_update = weight_updates[param]
        upd = mom * weight_update - lr * gparam
        updates[weight_update] = upd
        updates[param] = param + upd
    return updates
