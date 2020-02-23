
from flair.hyperparameter.param_selection import SequenceTaggerParamSelector, OptimizationValue
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings
from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import TaggedCorpus, MultiCorpus
from pathlib import Path
import flair, torch

flair.device = torch.device('cpu')
columns  = {0: 'text', 1: 'ner'}
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

custom_embedding = WordEmbeddings('../../glove/GloVe/vectors_converted_to_gensim.gensim')
#bert_embedding = BertEmbeddings('bert-embedding-files/')
word_embeddings = StackedEmbeddings([custom_embedding, WordEmbeddings('tr')])

search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[word_embeddings])
#search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128, 256, 512])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[256])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[2])
#search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2, 0.25])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[16])

param_selector = SequenceTaggerParamSelector(corpus=corpus, tag_type='ner', base_path='./results_tr_glove_embedding_learning_rate', max_epochs=10, training_runs=1, optimization_value=OptimizationValue.DEV_SCORE)

param_selector.optimize(search_space, max_evals=10)
