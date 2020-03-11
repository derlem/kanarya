# Author: Hasan Öztürk
# Usage: python extract_errors.py --flair_model_dirpath <path-to-flair-model>  --conll_dataset_dirpath <path-to-conll-data> --is_prob <include-probabilities-or-not>
# Example: python extract_errors.py python extract_errors.py --flair_model_dirpath  /opt/kanarya/resources/flair_models/bertcustom_2020-01-09_02 --conll_dataset_dirpath /home/hasan.ozturk/kanarya-github/kanarya/data/de-da-te-ta.10E-4percent.conll.84max.dev --is_prob True
# Note that we only take the last word w/ the de/da suffix into account.

import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
import flair, torch
import sys
import argparse
import copy


def get_labels(to_tagged_string):

		labels = []
		tokens = to_tagged_string.split()
		for token in tokens:
			if token == "<B-ERR>":
				# Remove the last "O" label !! I assumed that a prediction cannot start with a <B-ERR> tag
				del labels[-1]
				# Add <B-ERR> tag
				labels.append("B-ERR")
			else:
				labels.append("O")

		# Add "O" label for dot
		labels.append("O")
		return labels


def does_contain_deda(word):
	suffixes = ['de','da','te','ta']
	if word[-2:] in suffixes:
		return True
	else:
		return False

def div_dict(my_dict, dividend):
	for i in my_dict:
		my_dict[i] = my_dict[i] / dividend;

# Should take an object of Sentence class, not string as an argument
def get_prob(sentence, pos):

	deda_token = sentence.tokens[pos]
	label = deda_token.get_tag('ner') # Be careful, it returns 

	value = label.value

	if value == 'B-ERR':
		score = label.score
	else:
		score = 1 - label.score

	#print(str(deda_token) + ": " + str(label))
	return score # float


# Delete the last character of the preceding word
def perturb1(sentence, pos):

	# Preceding word
	perturbed_token_pos = max(0, pos-1)
	perturbed_sentence = copy.deepcopy(sentence)
	token = perturbed_sentence.tokens[perturbed_token_pos]
	if len(token.text) != 1:
		token.text = token.text[:-1]
	return perturbed_sentence


# Delete the first character of the preceding word
def perturb2(sentence, pos):

	# Preceding word
	perturbed_token_pos = max(0, pos-1)
	perturbed_sentence = copy.deepcopy(sentence)
	token = perturbed_sentence.tokens[perturbed_token_pos]
	if len(token.text) != 1:
		token.text = token.text[1:]
	return perturbed_sentence

# Capitalize the first character of the preceding word
def perturb3(sentence, pos):

	# Preceding word
	perturbed_token_pos = max(0, pos-1)
	perturbed_sentence = copy.deepcopy(sentence)
	token = perturbed_sentence.tokens[perturbed_token_pos]
	token.text = token.text.capitalize()
	return perturbed_sentence


perturbation_functions = [perturb1, perturb2, perturb3]

delta_p_dict = {}

sentence_count = 0

def create_dict():
	for func in perturbation_functions:
		delta_p_dict[func] = 0

def runner(params):

	flair.device = torch.device('cpu')

	#classifier_model = sys.argv[1]
	classifier_model = params["flair_model_dirpath"]
	conll_data = params["conll_dataset_dirpath"]
	#fname = sys.argv[2]

	classifier = SequenceTagger.load_from_file(classifier_model + '/best-model.pt')

	true_labels = []
	f = open(conll_data, "r")

	sentence = ""
	word_idx = 0 # 0 indexing
	deda_pos = 0
	for line in f:
		tokens = line.split()
		if(len(tokens) == 2):

			# Current word and label pair
			current_word = tokens[0] 
			current_label = tokens[1]

			# Currently we only take the last word which contains de/da in a sentence !!
			if does_contain_deda(current_word):
				deda_pos = word_idx


			word_idx += 1

			true_labels.append(current_label)

			# Check if it is the end of a sentence
			if(current_word == "."):
			
				# Construct the sentence
				sentence = sentence + current_word 

				# Make a prediction
				sentence = Sentence(sentence)
				classifier.predict(sentence)

				# Get the tagged string (does not include "O"s)
				tagged_string = sentence.to_tagged_string()

				# Get the labels for the sentence (does include "O"s)
				predicted_labels = get_labels(tagged_string)

				p0 = get_prob(sentence, deda_pos) 

				print(sentence)
				for func in perturbation_functions:
					perturbed_sentence = func(sentence, deda_pos)

					classifier.predict(perturbed_sentence)

					pi = get_prob(perturbed_sentence, deda_pos)

					delta_p = pi - p0

					print("Delta_p for " + str(func) + ": " + str(delta_p))
					delta_p_dict[func] += delta_p

				print("=================================")
				

				sentence = ""
				word_idx = 0
				global sentence_count
				sentence_count += 1
			else:
				# Construct the sentence
				# Careful: Sentence begins with a space !
				sentence = sentence + " " + current_word



def main():

    parser = argparse.ArgumentParser()

    best_bert_model = '/opt/kanarya/resources/flair_models/bertcustom_2020-01-09_02'
    test_set = '/home/hasan.ozturk/kanarya-github/kanarya/data/de-da-te-ta.10E-4percent.conll.84max.test'

    parser.add_argument("--conll_dataset_dirpath", default=test_set)
    parser.add_argument("--flair_model_dirpath", default=best_bert_model)

    args = parser.parse_args()

    conll_dataset_dirpath = args.conll_dataset_dirpath
    flair_model_dirpath = args.flair_model_dirpath

    params = {
    	"conll_dataset_dirpath": conll_dataset_dirpath,
    	"flair_model_dirpath": flair_model_dirpath,
    }

    create_dict()
    runner(params)

    global delta_p_dict

    print(delta_p_dict)
    print("Sentence Count: " + str(sentence_count))

    delta_p_dict = delta_p_dict / sentence_count
    div_dict(delta_p_dict)
    print(delta_p_dict)


if __name__ == "__main__":
    main()
