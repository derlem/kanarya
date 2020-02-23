from flair.data import TaggedCorpus, MultiCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BertEmbeddings
from typing import List
from flair.data import Dictionary
import flair, torch

columns = {0: 'text', 1: 'ner'}
data_folder = '../data'

corpus1: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                              train_file="de-da-te-ta.10E-4percent.conll.train",
                                                              test_file="de-da-te-ta.10E-4percent.conll.test",
                                                              dev_file="de-da-te-ta.10E-4percent.conll.dev")
corpus2: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                              train_file="de-da-te-ta.10E-4percent.conll.84max.train",
                                                              test_file="de-da-te-ta.10E-4percent.conll.84max.test",
                                                              dev_file="de-da-te-ta.10E-4percent.conll.84max.dev")

corpus = MultiCorpus([corpus1, corpus2])
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

#bert_embedding = BertEmbeddings('./bert-embedding-files')
#bert_embedding = BertEmbeddings('/home/hasan.ozturk/outputs/chunk1_vocab_29000_pretraining_output/model.ckpt-341000.data-00000-of-00001')
bert_embedding = BertEmbeddings('../outputs/bert_model/')

embedding_types: List[TokenEmbeddings] = [bert_embedding]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type, use_crf=True, use_rnn=True, rnn_layers=2)

from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

#trainer.train('./models/tr_glove2_word2vec_embedding_150_epochs_0.15_lr', learning_rate=0.15, mini_batch_size=16, max_epochs=150, checkpoint=True)
trainer.train('../outputs/tr_bert_embedding_1_epochs_0.15_lr', learning_rate=0.15, mini_batch_size=16, max_epochs=1, checkpoint=True)

