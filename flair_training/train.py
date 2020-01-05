import os
from typing import List


import flair
from flair.data import TaggedCorpus, MultiCorpus, Dictionary
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BertEmbeddings, CharacterEmbeddings
from flair.hyperparameter import SearchSpace, Parameter, SequenceTaggerParamSelector
from flair.hyperparameter.param_selection import OptimizationValue
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from hyperopt import hp
import numpy as np
import torch

data_folder = '../data'
tag_type = 'ner'


def load_corpus(data_folder, tag_type):
    columns = {0: 'text', 1: 'ner'}

    corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                                  train_file="de-da-te-ta.10E-4percent.conll.train",
                                                                  test_file="de-da-te-ta.10E-4percent.conll.test",
                                                                  dev_file="de-da-te-ta.10E-4percent.conll.dev")

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    return corpus, tag_dictionary


def create_embeddings(params):
    embedding_type = params["embedding_type"]
    assert embedding_type in ["bert", "flair", "char"]
    if embedding_type == "bert":
        bert_embedding = BertEmbeddings(params["bert_model_dirpath"])

        embedding_types: List[TokenEmbeddings] = [bert_embedding]
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    elif embedding_type == "flair":
        glove_embedding = WordEmbeddings('../../glove/GLOVE/GloVe/vectors.gensim')
        word2vec_embedding = WordEmbeddings('../../huawei_w2v/vector.gensim')

        # bert_embedding = BertEmbeddings('../bert_pretraining/pretraining_outputs/pretraining_output_batch_size_32')
        embedding_types: List[TokenEmbeddings] = [WordEmbeddings('tr'), glove_embedding, word2vec_embedding]
        # embedding_types: List[TokenEmbeddings] = [custom_embedding]
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    elif embedding_type == "char":
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=[CharacterEmbeddings()])
    else:
        embeddings = None

    return embeddings


def create_model(params, tag_dictionary):
    embeddings = create_embeddings(params["embedding_type"])
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=params["tag_type"],
                                            use_crf=True,
                                            use_rnn=True,
                                            rnn_layers=2)
    return tagger, embeddings


def select_hyperparameters(params, corpus):

    search_space = SearchSpace()

    embeddings = create_embeddings(params)
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embeddings])
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128, 256, 512])
    # search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[256])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.2, high=0.7)
    # search_space.add(Parameter.LEARNING_RATE, hp.loguniform, low=-np.log(0.00001), high=np.log(1.0))
    # search_space.add(Parameter.OPTIMIZER, hp.choice, options=[Parameter.NESTEROV])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[16, 32])

    if not os.path.exists("hyperparameter_search"):
        print("Creating the hyperparameter_search directory for hyperparameter selection process...")
        os.mkdir("hyperparameter_search")

    param_selector = SequenceTaggerParamSelector(corpus=corpus,
                                                 tag_type=params['tag_type'],
                                                 base_path=os.path.join("hyperparameter_search",
                                                                        params['model_output_dirpath']),
                                                 max_epochs=10,
                                                 training_runs=1,
                                                 optimization_value=OptimizationValue.DEV_SCORE)

    param_selector.optimize(search_space, max_evals=10)

    print("Now observe %s to decide on the best hyperparameters" % (os.path.join("hyperparameter_search",
                                                                        params['model_output_dirpath'],
                                                                        "param_selection.txt")))


def find_learning_rate(trainer, params):

    learning_rate_tsv = trainer.find_learning_rate(os.path.join("hyperparameter_search",
                                                                params['model_output_dirpath']),
                                                   'learning_rate_search_log.tsv')

    from flair.visual.training_curves import Plotter
    plotter = Plotter()
    plotter.plot_learning_rate(learning_rate_tsv)

def create_trainer(tagger, corpus):
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    return trainer


def train(params, tagger, corpus):
    trainer = create_trainer(tagger, corpus)

    # trainer.train('./models/tr_glove2_word2vec_embedding_150_epochs_0.15_lr', learning_rate=0.15, mini_batch_size=16, max_epochs=150, checkpoint=True)
    trainer.train(params["model_output_dirpath"],
                  learning_rate=params["learning_rate"],
                  mini_batch_size=params["mini_batch_size"],
                  max_epochs=params["max_epochs"],
                  checkpoint=True)


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", choices=["hyperparameter_search", "train"])
    parser.add_argument("--embedding_type", choices=["bert", "flair", "char"])
    parser.add_argument("--model_name", default="default_model_name")
    parser.add_argument("--bert_model_dirpath", default="../outputs/bert_model/")
    parser.add_argument("--model_output_dirpath", default="../outputs/tr_bert_embedding_1_epochs_0.15_lr")
    parser.add_argument("--learning_rate", default=0.05)
    parser.add_argument("--max_epochs", default=10)
    parser.add_argument("--mini_batch_size", default=16)



    args = parser.parse_args()

    command = args.command
    embedding_type = args.embedding_type
    model_name = args.model_name
    bert_model_dirpath = args.bert_model_dirpath
    model_output_dirpath = args.model_output_dirpath
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    mini_batch_size = args.mini_batch_size

    corpus, tag_dictionary = load_corpus(data_folder, tag_type)

    params = {
        "model_name": model_name,
        "embedding_type": embedding_type,
        "tag_type": tag_type,
        "bert_model_dirpath": bert_model_dirpath,
        "model_output_dirpath": model_output_dirpath,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "mini_batch_size": mini_batch_size
    }

    if command == "hyperparameter_search":
        select_hyperparameters(params, corpus)

        tagger, embeddings = create_model(params,
                                          tag_dictionary)

        trainer = create_trainer(tagger, corpus)

        find_learning_rate(trainer, params)

    elif command == "train":
        tagger, embeddings = create_model(params,
                                          tag_dictionary)

        train(params, tagger, corpus)


if __name__ == "__main__":
    main()