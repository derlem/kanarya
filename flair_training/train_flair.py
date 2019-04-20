from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
from flair.data import Dictionary
import flair, torch
flair.device = torch.device('cpu') 

columns = {0: 'text', 1: 'ner'}
data_folder = '../'
corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns, train_file="de-da-te-ta.10E-4percent.conll.train.txt", test_file="de-da-te-ta.10E-4percent.conll.test.txt", dev_file="de-da-te-ta.10E-4percent.conll.dev.txt")
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
#tag_dictionary: Dictionary = Dictionary.load('../vocab/m.model')
embedding_types: List[TokenEmbeddings] = [WordEmbeddings('glove'), WordEmbeddings('tr')]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type, use_crf=True)

from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('./models/glove_tr_embedding_0.1_learning_rate', learning_rate=0.1, mini_batch_size=32, max_epochs=150, checkpoint=True)


