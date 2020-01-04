
from pathlib import Path
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
import flair, torch
flair.device = torch.device('cpu')

columns = {0: 'text', 1: 'ner'}
data_folder = '../data'
corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                             train_file="de-da-te-ta.10E-4percent.conll.84max.train.txt",
                                                             test_file="de-da-te-ta.10E-4percent.conll.84max.test.txt",
                                                             dev_file="de-da-te-ta.10E-4percent.conll.84max.dev.txt")

trainer = ModelTrainer.load_from_checkpoint(Path('./models/example-ner-tr-embedding/checkpoint.pt'), 'SequenceTagger', corpus)

trainer.train('./models/example-ner-tr-embedding-continued', learning_rate=0.15, mini_batch_size=32, max_epochs=150, checkpoint=True)
