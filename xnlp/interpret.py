import logging
import os
import json
from pathlib import Path
from typing import List

import flair
from flair.data import Corpus, MultiCorpus, Dictionary, Sentence
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BertEmbeddings, CharacterEmbeddings
from flair.hyperparameter import SearchSpace, Parameter, SequenceTaggerParamSelector
from flair.hyperparameter.param_selection import OptimizationValue
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
from torch import Tensor

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

from flair_training.train import load_params, load_model, tag_type

import logging
logging.basicConfig(level=logging.INFO)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", choices=["interpret"], required=True)

    parser.add_argument("--model_output_dirpath", default=None)

    parser.add_argument("--data_folder", default="./data")

    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--loglevel", default="NOTSET", choices=["CRITICAL", "NOTSET", "INFO"])

    args = parser.parse_args()

    command = args.command

    model_output_dirpath = args.model_output_dirpath

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


def sort_and_get_first_k(logprobs, k=None):
    token_logprobs, token_ids = torch.sort(logprobs, dim=1, descending=True)
    if k is not None:
        return token_logprobs[:, :k], token_ids[:, :k]
    else:
        return token_logprobs, token_ids


def interpret_ner():

    # Load pre-trained model tokenizer (vocabulary)
    model_name = "dbmdz/bert-base-turkish-cased"
    # model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    "[CLS] Who was Jim Henson ? [SEP] Jim [MASK] was a puppeteer [SEP]"
    "[CLS] At a technical level, artificial intelligence seems to be the future of [MASK]"

    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()

    import sys

    line = sys.stdin.readline().strip()
    while line:
        segments_tensors, token_ids_in_sentence_tensor, tokens_sentence = get_token_ids_and_segment_ids(line, tokenizer)
        # Predict all tokens
        ltr_lm_loss, single_sentence_logprobs = get_ltr_lm_loss_and_token_logprobs_for_single_sentence(model,
                                                                                                       segments_tensors,
                                                                                                       token_ids_in_sentence_tensor)

        token_logprobs, token_ids = sort_and_get_first_k(single_sentence_logprobs, k=5)

        print(ltr_lm_loss)
        for word_idx in range(token_logprobs.shape[0]):
            tokens = tokenizer.convert_ids_to_tokens(token_ids[word_idx])
            print(word_idx, tokens_sentence[word_idx], tokens)

        print()
        line = sys.stdin.readline().strip()


def single_perturbation(text: str):
    import random
    random_char_idx = random.randint(0, len(text)-1)

    return text[:random_char_idx] + ' ' + text[random_char_idx+1:]


def perturbate():

    original_texts = ["", ""]
    original_texts[0] = "[CLS] Reuters’a konuşan ABD Dışişleri Bakanlığı’ndan bir yetkili, " \
                    "Suriye’de Türkiye’ye istihbarat paylaşımı ve teçhizat desteğinde " \
                    "bulunmayı planladıklarını söyledi."

    original_texts[1] = "[CLS] Reuters’a konuşan ABD Dışişleri Bakanlığı’ndan bir yetkili," \
                        " Suriye’de Türkiye’ye teçhizat desteğinde " \
                        "bulunmayı planladıklarını söyledi."

    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()

    for original_text in original_texts:
        print("ORIGINAL FIRST")
        calculate_ltr_lm_loss_of_a_single_sentence(model, original_text, tokenizer)
        n_perturbations = 0
        perturbed_text = single_perturbation(original_text)
        while perturbed_text and n_perturbations < 10:
            n_perturbations += 1
            calculate_ltr_lm_loss_of_a_single_sentence(model, perturbed_text, tokenizer)

            perturbed_text = single_perturbation(original_text)


def calculate_ltr_lm_loss_of_a_single_sentence(model, text, tokenizer):
    segments_tensors, token_ids_in_sentence_tensor, tokens_sentence = \
        get_token_ids_and_segment_ids(text, tokenizer)
    ltr_lm_loss, single_sentence_logprobs = \
        get_ltr_lm_loss_and_token_logprobs_for_single_sentence(model, segments_tensors, token_ids_in_sentence_tensor)
    print(text, ltr_lm_loss)
    return text, ltr_lm_loss


def get_ltr_lm_loss_and_token_logprobs_for_single_sentence(model, segments_tensors, token_ids_in_sentence_tensor):
    with torch.no_grad():
        outputs = model(token_ids_in_sentence_tensor,
                        lm_labels=token_ids_in_sentence_tensor,
                        token_type_ids=segments_tensors)
        # print(outputs)
        # print(len(outputs))
        ltr_lm_loss = outputs[0]
        token_prediction_logprobs = outputs[1]
        # print(token_prediction_logprobs.shape)
        single_sentence_logprobs = token_prediction_logprobs[0]
    return ltr_lm_loss, single_sentence_logprobs


def get_token_ids_and_segment_ids(text, tokenizer):
    tokenized_text = tokenizer.tokenize(text)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    try:
        sep_index = tokenized_text.index("[SEP]")
        segments_ids = [0] * (sep_index+1) + [1] * (len(indexed_tokens)-(sep_index+1))
    except ValueError as e:
        segments_ids = [0] * len(indexed_tokens)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return segments_tensors, tokens_tensor, tokenized_text


if __name__ == "__main__":
    # main()
    # interpret_ner()
    perturbate()

# # Load pre-trained model (weights)
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Set the model in evaluation mode to deactivate the DropOut modules
# # This is IMPORTANT to have reproducible results during evaluation!
# model.eval()
#
# # If you have a GPU, put everything on cuda
# # tokens_tensor = tokens_tensor.to('cuda')
# # segments_tensors = segments_tensors.to('cuda')
# # model.to('cuda')
#
# # Predict hidden states features for each layer
# with torch.no_grad():
#     # See the models docstrings for the detail of the inputs
#     outputs = model(tokens_tensor, token_type_ids=segments_tensors)
#     # Transformers models always output tuples.
#     # See the models docstrings for the detail of all the outputs
#     # In our case, the first element is the hidden state of the last layer of the Bert model
#     encoded_layers = outputs[0]
# # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
# assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)
