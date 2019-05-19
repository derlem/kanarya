import gensim
import sys

word2vec_embedding_path = sys.argv[1]
converted_embedding_path = sys.argv[2]

word_vectors = gensim.models.KeyedVectors.load_word2vec_format(word2vec_embedding_path, binary=False)
word_vectors.save(converted_embedding_path)
