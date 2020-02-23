from flair.data import TaggedCorpus, MultiCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BertEmbeddings
from typing import List
from flair.data import Dictionary
import flair, torch
flair.device = torch.device('cpu') 

columns = {0: 'text', 1: 'ner'}
data_folder = '../data'
corpus1: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                              train_file="de-da-te-ta.10E-4percent.conll.train.txt",
                                                              test_file="de-da-te-ta.10E-4percent.conll.test.txt",
                                                              dev_file="de-da-te-ta.10E-4percent.conll.dev.txt")
corpus2: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                              train_file="de-da-te-ta.10E-4percent.conll.84max.train.txt",
                                                              test_file="de-da-te-ta.10E-4percent.conll.84max.test.txt",
                                                              dev_file="de-da-te-ta.10E-4percent.conll.84max.dev.txt")
corpus = MultiCorpus([corpus1, corpus2])
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
#tag_dictionary: Dictionary = Dictionary.load('../vocab/m.model')

glove_embedding = WordEmbeddings('../../glove/GLOVE/GloVe/vectors.gensim')
word2vec_embedding = WordEmbeddings('../../huawei_w2v/vector.gensim')

#bert_embedding = BertEmbeddings('../bert_pretraining/pretraining_outputs/pretraining_output_batch_size_32')
embedding_types: List[TokenEmbeddings] = [WordEmbeddings('tr'), glove_embedding, word2vec_embedding]
#embedding_types: List[TokenEmbeddings] = [custom_embedding]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True,
                                        use_rnn=True,
                                        rnn_layers=2)

from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('./models/tr_glove2_word2vec_embedding_150_epochs_0.1_lr',
              learning_rate=0.1,
              mini_batch_size=16,
              max_epochs=150,
              checkpoint=True)


