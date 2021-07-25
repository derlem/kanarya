import gensim
import sys

glove_embedding_path = sys.argv[1]
converted_embedding_path = sys.argv[2]

gensim.scripts.glove2word2vec.glove2word2vec(glove_embedding_path, 'glove_to_word2vec_temp_file.txt')

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('glove_to_word2vec_temp_file.txt', binary=False)
word_vectors.save(converted_embedding_path)
