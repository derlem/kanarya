import os
import json
from typing import List


import flair
from flair.data import TaggedCorpus, MultiCorpus, Dictionary
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BertEmbeddings, CharacterEmbeddings
from flair.hyperparameter import SearchSpace, Parameter, SequenceTaggerParamSelector
from flair.hyperparameter.param_selection import OptimizationValue
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
from hyperopt import hp
import numpy as np
import torch

data_folder = '../data'
tag_type = 'ner'


def load_corpus(data_folder, tag_type):
    columns = {0: 'text', 1: 'ner'}

    corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                                  train_file="de-da-te-ta.10E-4percent.conll.84max.train",
                                                                  test_file="de-da-te-ta.10E-4percent.conll.84max.test",
                                                                  dev_file="de-da-te-ta.10E-4percent.conll.84max.dev")

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    return corpus, tag_dictionary


def create_embeddings(params):
    embedding_type = params["embedding_type"]
    assert embedding_type in ["bert", "flair", "char"]
    if embedding_type == "bert":
        bert_embedding = BertEmbeddings(params["bert_model_dirpath_or_name"])

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
    embeddings = create_embeddings(params)
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=params["tag_type"],
                                            use_crf=True,
                                            use_rnn=True,
                                            rnn_layers=2)
    return tagger, embeddings


def load_model(model_output_dirpath):
    final_model_filepath = os.path.join(model_output_dirpath, "final-model.pt")
    tagger = SequenceTagger.load_from_file(final_model_filepath)
    return tagger


def select_hyperparameters(params, corpus):

    search_space = SearchSpace()

    embeddings = create_embeddings(params)
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embeddings])
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128, 256, 512])
    # search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[256])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
    search_space.add(Parameter.DROPOUT, hp.choice, options=[0.3, 0.4, 0.5, 0.6, 0.7])
    # search_space.add(Parameter.LEARNING_RATE, hp.loguniform, low=-np.log(0.00001), high=np.log(1.0))
    # search_space.add(Parameter.OPTIMIZER, hp.choice, options=[Parameter.NESTEROV])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[16])

    if not os.path.exists("hyperparameter_search"):
        print("Creating the hyperparameter_search directory for hyperparameter selection process...")
        os.mkdir("hyperparameter_search")

    print("Downsampling the training set to %10 of the original...")
    corpus.downsample(percentage=0.1, only_downsample_train=True)

    param_selector = SequenceTaggerParamSelector(corpus=corpus,
                                                 tag_type=params['tag_type'],
                                                 base_path=os.path.join("hyperparameter_search",
                                                                        params['model_output_dirpath']),
                                                 max_epochs=5,
                                                 training_runs=3,
                                                 evaluation_metric=EvaluationMetric.MICRO_F1_SCORE,
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


def save_params(params, filepath):

    with open(filepath, "w") as f:
        json.dump(params, f)


def load_params(filepath):

    with open(filepath, "r") as f:
        params = json.loads("\n".join(f.readlines()))

    return params


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", choices=["hyperparameter_search", "train", "evaluate"], required=True)
    parser.add_argument("--embedding_type", choices=["bert", "flair", "char"])
    parser.add_argument("--model_name", default="default_model_name")
    parser.add_argument("--bert_model_dirpath_or_name", default="bert-base-multilingual-cased")
    parser.add_argument("--model_output_dirpath", default=None)
    parser.add_argument("--learning_rate", default=0.05)
    parser.add_argument("--max_epochs", default=10)
    parser.add_argument("--mini_batch_size", default=16)

    parser.add_argument("--data_folder", default="./data")

    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])

    args = parser.parse_args()

    command = args.command
    embedding_type = args.embedding_type
    model_name = args.model_name
    bert_model_dirpath_or_name = args.bert_model_dirpath_or_name # "../outputs/bert_model/"

    model_output_dirpath = args.model_output_dirpath
    if model_output_dirpath is None and command != "evaluate":
        if not os.path.exists("./models"):
            os.mkdir("./models")
        model_output_dirpath = "./models/%s" % model_name
        if not os.path.exists(model_output_dirpath):
            os.mkdir(model_output_dirpath)

    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    mini_batch_size = args.mini_batch_size
    data_folder = args.data_folder
    device = args.device

    if device != "gpu" and device == "cpu":
        flair.device = torch.device('cpu')

    corpus, tag_dictionary = load_corpus(data_folder, tag_type)

    params = {
        "model_name": model_name,
        "embedding_type": embedding_type,
        "tag_type": tag_type,
        "bert_model_dirpath_or_name": bert_model_dirpath_or_name,
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

        if not os.path.exists(params["model_output_dirpath"]):
            os.mkdir(params["model_output_dirpath"])

        tag_dictionary.save(os.path.join(params["model_output_dirpath"], "tag_dictionary.pickle"))

        save_params(params,
                    os.path.join(params["model_output_dirpath"], "params.json"))

        train(params, tagger, corpus)
    elif command == "evaluate":
        try:
            tag_dictionary: Dictionary = Dictionary.load_from_file(os.path.join(model_output_dirpath,
                                                                            "tag_dictionary.pickle"))
        except FileNotFoundError:
            print("tag_dictionary is not found at %s" % os.path.join(model_output_dirpath,
                                                                            "tag_dictionary.pickle"))
        params = load_params(os.path.join(model_output_dirpath,
                                          "params.json"))
        tagger: SequenceTagger = load_model(model_output_dirpath)

        trainer = create_trainer(tagger, corpus)
        trainer.evaluate(tagger, corpus.test, eval_mini_batch_size=16,
                         out_path=os.path.join(params["model_output_dirpath"], "evaluation.txt"))


if __name__ == "__main__":
    main()