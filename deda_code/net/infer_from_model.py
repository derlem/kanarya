import _pickle as cPickle
import numpy
from reader import io_util, word2vec_reader
import file_path
from config import *


def file_list_gen(X_test, Y_test, X_test_w, nnets_file_name):
    # print('Starting generation')

    nnet_model = cPickle.load(open(nnets_file_name, 'rb'))

    file_number = len(X_test)
    print()
    print()
    for i in range(file_number):  # file_number
        test_set_x = X_test[i]
        predicted_parameter = nnet_model.parameter_prediction(test_set_x)

        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        w_list = X_test_w[i]
        tar_list = Y[i]
        for j in range(len(predicted_parameter)):
            pred = 'O' if predicted_parameter[j] < 0.5 else 'B-ERR'
            tar = 'O' if tar_list[j] < 0.5 else 'B-ERR'
            print('%s %s %s' % (w_list[j], tar, pred))
        print()


X, Y, X_w = word2vec_reader.prep_deda_data(
    file_path.test_path, file_path.used_embed_path)

# lim = 1
# X = X[0:lim]
# Y = Y[0:lim]
# X_w = X_w[0:lim]
# test_set_x = io_util.make_shared(X_test[0], 'test_x')
# test_set_y = io_util.make_shared(Y_test[0], 'test_y')

# print('Reloading model')
net_model = cPickle.load(open(reload_model_path, 'rb'))
file_list_gen(X, Y, X_w, reload_model_path)
