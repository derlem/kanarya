import numpy as np
import random
from reader import prep
import theano


def load_binary_file(file_name, dimension, dtype=np.float32):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=dtype)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0, 'specified dimension %s not compatible with data(%s)'\
                                                    % (dimension, features.size)
    features = features[:dimension * (features.size / dimension)]
    features = features.reshape((-1, dimension))
    return features


def save_binary_file(data, output_file_name):
    data = np.array(data, 'float32')
    fid = open(output_file_name, 'wb')
    data.tofile(fid)
    fid.close()


def load_binary_file_frame(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0, 'specified dimension %s not compatible with data' % dimension
    frame_number = features.size / dimension
    features = features[:dimension * frame_number]
    features = features.reshape((-1, dimension))
    return features, frame_number


def get_seq_len(file_list, reader, **kwargs):
    seq_len_list = []
    for file_path in file_list:
        feat = reader(file_path, **kwargs)
        seq_len, _ = feat.shape
        seq_len_list.append(seq_len)
    return seq_len_list


def make_shared(data_set, data_name):
    data_set = theano.shared(np.asarray(data_set, dtype=theano.config.floatX), name=data_name,
                             borrow=True)  # @UndefinedVariable

    return data_set


class Feeder(object):

    def __init__(self, item_list, batch_size, max_len, shuffle=False, seed=123):
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.max_len = max_len
        if shuffle:
            random.seed(self.seed)
            random.shuffle(item_list)
        self.item_list = item_list
        self.len = len(item_list)
        if self.batch_size > self.len:
            raise ValueError('Batch size %d cannot be greater than the length of item list %d'
                             % (self.batch_size, self.len))
        self.cur = 0

    def reset(self):
        self.cur = 0
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.item_list)

    def is_finished(self):
        return self.len - self.cur < self.batch_size

    def get_next(self):
        item = self.item_list[self.cur]
        self.cur += 1
        return item

    def get_next_batch(self):
        batch_list = self.item_list[self.cur:self.cur+self.batch_size]
        self.cur += self.batch_size
        return batch_list

    def load_next(self, read, **kwargs):
        return read(self.get_next(), **kwargs)

    def load_next_batch(self, read, **kwargs):
        b = None
        for i in range(self.batch_size):
            feat = self.load_next(read, **kwargs)
            feat_padded = prep.pad_zero(feat, self.max_len)
            if b is None:
                r, c = feat_padded.shape
                b = np.zeros((self.batch_size, r, c))
                b[i, :, :] = feat_padded
            else:
                b[i, :, :] = feat_padded
        return b
