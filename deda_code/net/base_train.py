import _pickle as cPickle
import numpy as np
from reader import file_name, io_util, word2vec_reader
import file_path
from config import *
import time
from net import model
import sys
import theano
import math
import scipy.io as sio
from random import shuffle
import sys


X_tr, Y_tr, _ = word2vec_reader.prep_deda_data(
    file_path.train_path, file_path.used_embed_path)
X_val, Y_val, _ = word2vec_reader.prep_deda_data(
    file_path.dev_path, file_path.used_embed_path)


train_set_x = io_util.make_shared(X_tr[0], 'train_x')
train_set_y = io_util.make_shared(Y_tr[0], 'train_y')

valid_set_x = io_util.make_shared(X_val[0], 'val_x')
valid_set_y = io_util.make_shared(Y_val[0], 'val_y')

if not reload_model:
    print('Building model')
    net_model = model.BaseModel(in_dim, hidden_layer_size, out_dim, 
                                hidden_layer_type, 
                                output_type='SIGMOID', 
                                dropout_rate=drp_rate)
else:
    print('Reloading model')
    net_model = cPickle.load(open(reload_model_path, 'rb'))

tr_step, val_step = net_model.build_steps((train_set_x, train_set_y), 
                                          (valid_set_x, valid_set_y), 
                                          opt=opt)
if trust_input:
    val_step.trust_input = True

epoch_time = []
epoch_train_loss = []
epoch_val_loss = []
epoch_acc = []
stat = {}
start_time = time.time()

best_dnn_model = net_model
best_validation_loss = sys.float_info.max
previous_loss = sys.float_info.max

early_stop = 0
e = 0

previous_finetune_lr = lr
print('Starting training')
while e < epoch:
    e = e + 1

    current_momentum = momentum
    current_finetune_lr = lr
    if e <= warmup_epoch or not half_lr:
        current_finetune_lr = lr
        current_momentum = warmup_momentum
    else:
        current_finetune_lr = previous_finetune_lr * 0.5

    previous_finetune_lr = current_finetune_lr

    train_error = []
    sub_start_time = time.time()

    p = np.random.permutation(len(X_tr))
    X_tr = [X_tr[i] for i in p]
    Y_tr = [Y_tr[i] for i in p]

    for i in range(len(X_tr)):

        temp_train_set_x = X_tr[i]
        temp_train_set_y = Y_tr[i]

        train_set_x.set_value(np.asarray(temp_train_set_x,
                                         dtype=theano.config.floatX), 
                                         borrow=True)  # @UndefinedVariable
        train_set_y.set_value(np.asarray(temp_train_set_y,
                                         dtype=theano.config.floatX), 
                                         borrow=True)  # @UndefinedVariable

        this_train_error = tr_step(current_finetune_lr, current_momentum)

        train_error.append(this_train_error)

    print('calculating validation loss')
    validation_losses = []
    false_pred = 0.
    for i in range(len(X_val)):
        temp_valid_set_x = X_val[i]
        temp_valid_set_y = Y_val[i]

        valid_set_x.set_value(np.asarray(temp_valid_set_x, dtype=theano.config.floatX),
                              borrow=True)  # @UndefinedVariable
        valid_set_y.set_value(np.asarray(temp_valid_set_y, dtype=theano.config.floatX),
                              borrow=True)  # @UndefinedVariable

        this_valid_loss, diff = val_step()
        if diff > 0:
            false_pred += 1
        
        
        validation_losses.append(this_valid_loss)

    val_acc = (len(X_val) - false_pred)/len(X_val)
    this_validation_loss = np.mean(validation_losses)

    this_train_valid_loss = np.mean(np.asarray(train_error))

    sub_end_time = time.time()
    epoch_train_loss.append(this_train_valid_loss)
    epoch_val_loss.append(this_validation_loss)
    epoch_time.append((sub_end_time - sub_start_time))
    epoch_acc.append(val_acc)
    print('epoch %i, validation error %f, \
        train error %f time spent %.2f' % 
        (e, this_validation_loss, this_train_valid_loss, 
        (sub_end_time - sub_start_time)))
    print('validation acc: %f' % val_acc)

    if this_validation_loss < best_validation_loss:
        if save_model:
            cPickle.dump(best_dnn_model, open(save_file_path, 'wb'))

        best_dnn_model = net_model
        best_validation_loss = this_validation_loss

    if this_validation_loss >= previous_loss:
        print('validation loss increased')

        early_stop += 1

    if e > 15 and early_stop > patience:
        print('stopping early')
        break

    if math.isnan(this_validation_loss):
        break

    previous_loss = this_validation_loss
    stat['val_acc'] = epoch_acc
    stat['time'] = epoch_time
    stat['train_loss'] = epoch_train_loss
    stat['val_loss'] = epoch_val_loss
    stat_path = save_file_path + '_stat' + '.mat'
    sio.savemat(stat_path, stat)
end_time = time.time()

print('overall  training time: %.2fm validation error %f' % ((end_time - start_time) / 60., best_validation_loss))

