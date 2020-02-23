import logging
import os
import json
from pathlib import Path
from typing import List


import flair
from flair.data import TaggedCorpus, MultiCorpus, Dictionary, Sentence
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
from torch.optim import Adam, SGD, AdamW

log = logging.getLogger('flair')

data_folder = '../data'
tag_type = 'ner'

optimizers_dict = {"sgd": SGD, "adam": Adam, "adamw": AdamW}


def load_standard_corpus(data_folder, tag_type):

    files = {"train_file": "de-da-te-ta.10E-4percent.conll.84max.train",
             "dev_file": "de-da-te-ta.10E-4percent.conll.84max.dev",
             "test_file": "de-da-te-ta.10E-4percent.conll.84max.test"}
    if "10E-3percent" in data_folder:
        files = {"train_file": "de-da-te-ta.10E-3percent.conll.84max.train",
                 "dev_file": "de-da-te-ta.10E-4percent.conll.84max.dev",
                 "test_file": "de-da-te-ta.10E-4percent.conll.84max.test"}
    for key, value in files.items():
        files[key] = os.path.abspath(os.path.join(data_folder, value))

    corpus = load_specific_corpus(data_folder, files)

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    log.info(corpus.obtain_statistics(tag_type=tag_type))

    return corpus, tag_dictionary


def load_specific_corpus(dirname, files):

    columns = {0: 'text', 1: 'ner'}
    if files['train_file']:
        train_file = os.path.basename(files['train_file'])
    else:
        train_file = None
    if files['dev_file']:
        dev_file = os.path.basename(files['dev_file'])
    else:
        dev_file = None
    if files['test_file']:
        test_file = os.path.basename(files['test_file'])
    else:
        test_file = None

    corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(dirname,
                                                                 columns,
                                                                 train_file=train_file,
                                                                 dev_file=dev_file,
                                                                 test_file=test_file)
    log.info(corpus)

    return corpus


def create_embeddings(params):
    embedding_type = params["embedding_type"]
    assert embedding_type in ["bert", "flair", "char"]
    if embedding_type == "bert":
        bert_embedding = BertEmbeddings(params["bert_model_dirpath_or_name"],
                                        pooling_operation="mean")

        embedding_types: List[TokenEmbeddings] = [bert_embedding]
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    elif embedding_type == "flair":
        glove_embedding = WordEmbeddings('/opt/kanarya/glove/GLOVE/GloVe/vectors.gensim')
        word2vec_embedding = WordEmbeddings('/opt/kanarya/huawei_w2v/vector.gensim')
        fast_text_embedding = WordEmbeddings('tr')
        char_embedding = CharacterEmbeddings()

        # bert_embedding = BertEmbeddings('../bert_pretraining/pretraining_outputs/pretraining_output_batch_size_32')
        embedding_types: List[TokenEmbeddings] = [fast_text_embedding, glove_embedding, word2vec_embedding, char_embedding]
        # embedding_types: List[TokenEmbeddings] = [custom_embedding]
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    elif embedding_type == "char":
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=[CharacterEmbeddings()])
    else:
        embeddings = None

    return embeddings


def create_model(params, tag_dictionary):
    embeddings = create_embeddings(params)
    tagger: SequenceTagger = SequenceTagger(hidden_size=params["hidden_size"],
                                            dropout=params["dropout"],
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=params["tag_type"],
                                            use_crf=True,
                                            use_rnn=True,
                                            rnn_layers=2)
    return tagger, embeddings


def load_model(model_output_dirpath):
    final_model_filepath = os.path.join(model_output_dirpath, "final-model.pt")
    tagger: SequenceTagger = SequenceTagger.load_from_file(final_model_filepath)
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
                                                   'learning_rate_search_log.tsv',
                                                   iterations=400,
                                                   stop_early=False,
                                                   mini_batch_size=16)

    from flair.visual.training_curves import Plotter
    plotter = Plotter()
    plotter.plot_learning_rate(learning_rate_tsv)


def create_trainer(tagger, corpus, optimizer = SGD):
    trainer: ModelTrainer = ModelTrainer(tagger,
                                         corpus,
                                         optimizer=optimizer)
    return trainer


def train(params, tagger, corpus):
    if tagger is not None:
        trainer = create_trainer(tagger, corpus)
    else:
        print("Resuming training")
        trainer = ModelTrainer.load_from_checkpoint(Path(os.path.join(params["model_output_dirpath"],
                                                                 "checkpoint.pt")),
                                                    'SequenceTagger',
                                                    corpus)

    # trainer.train('./models/tr_glove2_word2vec_embedding_150_epochs_0.15_lr', learning_rate=0.15, mini_batch_size=16, max_epochs=150, checkpoint=True)
    trainer.train(params["model_output_dirpath"],
                  learning_rate=params["learning_rate"],
                  mini_batch_size=params["mini_batch_size"],
                  max_epochs=params["max_epochs"],
                  patience=3,
                  anneal_factor=0.5,
                  anneal_against_train_loss=True,
                  anneal_with_restarts=True,
                  checkpoint=True,
                  embeddings_in_memory=True)


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
    parser.add_argument("--command", choices=["hyperparameter_search",
                                              "find_learning_rate",
                                              "train",
                                              "resume_train",
                                              "evaluate",
                                              "evaluate_from_stdin",
                                              "evaluate_zor_cumleler"], required=True)
    parser.add_argument("--embedding_type", choices=["bert", "flair", "char"])
    parser.add_argument("--model_name", default="default_model_name")
    parser.add_argument("--bert_model_dirpath_or_name", default="bert-base-multilingual-cased")
    parser.add_argument("--model_output_dirpath", default=None)
    parser.add_argument("--other_test_file_for_evaluation", default=None)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.05, type=float)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--mini_batch_size", default=16, type=int)

    parser.add_argument("--data_folder", default="./data")

    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--loglevel", default="NOTSET", choices=["CRITICAL", "NOTSET", "INFO"])

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

    other_test_file_for_evaluation = args.other_test_file_for_evaluation

    optimizer = args.optimizer
    hidden_size = args.hidden_size
    dropout = args.dropout
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    mini_batch_size = args.mini_batch_size
    data_folder = args.data_folder
    device = args.device
    loglevel = args.loglevel

    log.setLevel(loglevel)
    if command in ["evaluate", "evaluate_from_stdin", "evaluate_zor_cumleler"]:
        log.setLevel("CRITICAL")

    if device != "gpu" and device == "cpu":
        flair.device = torch.device('cpu')

    if other_test_file_for_evaluation is None:
        corpus, tag_dictionary = load_standard_corpus(data_folder, tag_type)
    else:
        dirname = os.path.dirname(other_test_file_for_evaluation)
        corpus = load_specific_corpus(dirname, {"train_file": None,
                                                "dev_file": None,
                                                "test_file": os.path.basename(other_test_file_for_evaluation)})
        # tag dictionary will be loaded from the model
        tag_dictionary = None

    assert corpus, "corpus should be loaded"

    params = {
        "model_name": model_name,
        "embedding_type": embedding_type,
        "tag_type": tag_type,
        "bert_model_dirpath_or_name": bert_model_dirpath_or_name,
        "model_output_dirpath": model_output_dirpath,
        "optimizer": optimizer,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "mini_batch_size": mini_batch_size
    }

    if command in ["hyperparameter_search", "find_learning_rate"]:

        if not os.path.exists("hyperparameter_search"):
            print("Creating the hyperparameter_search directory for hyperparameter selection process...")
            os.mkdir("hyperparameter_search")

        if command == "hyperparameter_search":
            select_hyperparameters(params, corpus)

        tagger, embeddings = create_model(params,
                                          tag_dictionary)

        trainer = create_trainer(tagger, corpus, optimizer=optimizers_dict[optimizer])

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
    elif command == "resume_train":
        params = load_params(os.path.join(model_output_dirpath,
                                          "params.json"))
        train(params, None, corpus)
    elif command == "evaluate":
        try:
            tag_dictionary: Dictionary = Dictionary.load_from_file(os.path.join(model_output_dirpath,
                                                                            "tag_dictionary.pickle"))
        except FileNotFoundError:
            log.warning("WARN: tag_dictionary is not found at %s" % os.path.join(model_output_dirpath,
                                                                            "tag_dictionary.pickle"))
        params = load_params(os.path.join(model_output_dirpath,
                                          "params.json"))
        tagger: SequenceTagger = load_model(model_output_dirpath)

        trainer = create_trainer(tagger, corpus)
        if other_test_file_for_evaluation:
            out_path = other_test_file_for_evaluation + ".evaluation"
        else:
            out_path = os.path.join(params["model_output_dirpath"], "evaluation.txt")
        metric, eval_loss = trainer.evaluate(tagger, corpus.test, eval_mini_batch_size=16,
                         out_path=Path(out_path))

        print(metric.to_tsv())
    elif command == "evaluate_from_stdin":
        try:
            tag_dictionary: Dictionary = Dictionary.load_from_file(os.path.join(model_output_dirpath,
                                                                                "tag_dictionary.pickle"))
        except FileNotFoundError:
            print("WARN: tag_dictionary is not found at %s" % os.path.join(model_output_dirpath,
                                                                           "tag_dictionary.pickle"))
        params = load_params(os.path.join(model_output_dirpath,
                                          "params.json"))
        tagger: SequenceTagger = load_model(model_output_dirpath)

        import sys
        line = sys.stdin.readline().strip()
        while line:
            sentence = Sentence(line, use_tokenizer=True)
            tagger.predict([sentence], mini_batch_size=1, verbose=True)
            # print(sentence)
            # print(sentence.to_tagged_string())
            for token in sentence.tokens:
                tag = token.get_tag(tag_type)
                print(token.text, 'O', tag.value, tag.score)
            print()
            line = sys.stdin.readline().strip()

    elif command == "evaluate_zor_cumleler":
        try:
            tag_dictionary: Dictionary = Dictionary.load_from_file(os.path.join(model_output_dirpath,
                                                                                "tag_dictionary.pickle"))
        except FileNotFoundError:
            log.warning("WARN: tag_dictionary is not found at %s" % os.path.join(model_output_dirpath,
                                                                           "tag_dictionary.pickle"))
        params = load_params(os.path.join(model_output_dirpath,
                                          "params.json"))
        tagger: SequenceTagger = load_model(model_output_dirpath)

        metrics = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        with open("data/zor_cumleler.txt") as f:
            line = f.readline().strip()
            while line:
                sentence = Sentence(line, use_tokenizer=True)
                tagger.predict([sentence], mini_batch_size=1, verbose=False)
                if len(sentence.get_spans(tag_type)) > 0:
                    metrics["TP"] += 1
                else:
                    metrics["FN"] += 1
                line = f.readline().strip()
        print(metrics)


if __name__ == "__main__":
    main()