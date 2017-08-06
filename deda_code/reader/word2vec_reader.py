import file_path
import numpy as np


def extract_used_words(word2vec_path, train_path, dev_path):

    word_dim = 300

    word_d = {}
    unknown_emb = np.random.rand(word_dim,)
    # print(unknown_emb.shape)

    with open(train_path) as f:
        for line in f:
            if line == '\n':
                continue
            word = str(line.split()[0])
            # print(word)
            word_d[word] = unknown_emb

    with open(dev_path) as f:
        for line in f:
            if line == '\n':
                continue
            word = str(line.split()[0])
            # print(word)
            word_d[word] = unknown_emb

    # print('unique words: %d' % len(word_d))

    known_count = 0
    with open(word2vec_path) as f:
        f.readline()
        idx = 0
        for line in f:
            idx += 1
            # if idx % 10000 == 0:
            #    print(idx)
            elems = line.split()
            word = str(elems[0])
            if word in word_d:
                known_count += 1
                print(line.strip())

    # print('done. known count: %d' % known_count)


def get_word_embeds(embed_path):
    word_d = {}

    with open(embed_path) as f:
        for line in f:
            if line == '\n':
                continue
            tok, *vec = line.split()
            word_d[tok] = np.asarray(vec, dtype=np.float32)
    return word_d


def prep_deda_data(data_path, used_embed_path):
    word_dim = 300
    word_d = get_word_embeds(used_embed_path)
    unknown_emb = np.random.rand(word_dim)
    X = []
    Y = []
    x = []
    y = []
    X_words = []
    x_w = []
    with open(data_path) as f:
        f.readline()
        f.readline()
        for line in f:
            if line == '\n':
                x_sample = np.asarray(x, dtype=np.float32)
                x = []
                y_sample = np.asarray(y, dtype=np.float32)
                y = []
                X.append(x_sample)
                Y.append(y_sample.reshape((-1, 1)))
                X_words.append(x_w)
                x_w = []
                continue
            word, target = line.split()
            if target == 'B-ERR':
                target = 1
            elif target == 'O':
                target = 0
            else:
                raise ValueError('%s is not a known target value' % target)
            if word in word_d:
                word_emb = word_d[word]
            else:
                word_emb = unknown_emb
            x.append(word_emb)
            y.append(target)
            x_w.append(word)
    return X, Y, X_words

if __name__ == "__main__":
    print('word2vec_reader main')
    # extract_used_words(file_path.word2vec_path, file_path.train_path, file_path.dev_path)
    # get_word_embeds(file_path.used_embed_path)
    X, Y = prep_deda_data(file_path.train_path, file_path.used_embed_path)
    print(X[0].shape)
    print(len(X))
    
