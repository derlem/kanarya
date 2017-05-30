import numpy as np


def compute_mean(f_list, dim, read_fn, **kwargs):

    mean_vector = np.zeros((1, dim))
    total_frame_number = 0

    for filename in f_list:
        v, v_frame = read_fn(filename, **kwargs)

        mean_vector += np.reshape(np.sum(v, axis=0), (1, dim))

        total_frame_number += v_frame

    mean_vector /= float(total_frame_number)

    return mean_vector


def compute_std(f_list, mean_vector, dim, read_fn, **kwargs):

    std_vector = np.zeros((1, dim))
    total_frame_number = 0

    for filename in f_list:
        v, v_frame = read_fn(filename, **kwargs)

        mean_matrix = np.tile(mean_vector, (v_frame, 1))

        std_vector += np.reshape(np.sum((v - mean_matrix) ** 2, axis=0), (1, dim))

        total_frame_number += v_frame

    std_vector /= float(total_frame_number)

    std_vector = std_vector ** 0.5

    return std_vector


def normalize(feature, mean_vector, std_vector):
    frame, _ = feature.shape

    mean_matrix = np.tile(mean_vector, (frame, 1))
    std_matrix = np.tile(std_vector, (frame, 1))

    return (feature - mean_matrix) / std_matrix


def normalize_list_save(in_f_list, out_f_list, mean_vector, std_vector, read_fn, write_fn, **kwargs):
    assert len(in_f_list) == len(out_f_list), 'length of input list and output list doesn\'t match'

    for i in xrange(len(in_f_list)):
        in_file = in_f_list[i]
        out_file = out_f_list[i]
        raw_feature = read_fn(in_file, **kwargs)

        norm_v = normalize(raw_feature, mean_vector, std_vector)

        write_fn(norm_v, out_file)


def add_noise(feature, mu=0, sigma=0.1):
    r, c = feature.shape
    return feature + np.random.normal(mu, sigma, (r, c))


def pad_zero(feat, max_len):
    seq_len, _ = feat.shape
    pad_len = max_len - seq_len
    # npad is a tuple of (n_before, n_after) for each dimension
    npad = ((0, pad_len), (0, 0))
    return np.pad(feat, npad, mode='constant', constant_values=0)
