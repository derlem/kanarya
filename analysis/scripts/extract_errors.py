# Author: Hasan Öztürk
# Usage: python extract_errors.py --flair_model_dirpath <path-to-flair-model>  --conll_dataset_dirpath <path-to-conll-data> --is_prob <include-probabilities-or-not>
# Example: python extract_errors.py python extract_errors.py --flair_model_dirpath  /opt/kanarya/resources/flair_models/bertcustom_2020-01-09_02 --conll_dataset_dirpath /home/hasan.ozturk/kanarya-github/kanarya/data/de-da-te-ta.10E-4percent.conll.84max.dev --is_prob True

import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
import flair, torch
import sys
import argparse

type1_errors = []
type2_errors = []

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

def classify(params):

	flair.device = torch.device('cpu')

	#classifier_model = sys.argv[1]
	classifier_model = params["flair_model_dirpath"]
	conll_data = params["conll_dataset_dirpath"]
	is_prob = params["is_prob"]
	#fname = sys.argv[2]

	classifier = SequenceTagger.load_from_file(classifier_model + '/best-model.pt')

	true_labels = []
	f = open(conll_data, "r")

	sentence = ""
	for line in f:
		tokens = line.split()
		if(len(tokens) == 2):

			# Current word and label pair
			current_word = tokens[0]
			current_label = tokens[1]

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
			
				# if there is an error in our prediction
				if predicted_labels != true_labels:
					# The model could not find the error when there is one. (False negative)
					if 'B-ERR' in true_labels:
						type2_errors.append('* ' + tagged_string)

						if is_prob:
							for token in sentence.tokens:
								prob = token.get_tag('ner')
								prob_s = '{:<30s}{:<30s}'.format(str(token),str(prob))
								type2_errors.append(prob_s)
							type2_errors.append('\n')

					# The model find an error when there is not (False positive)
					else:
						type1_errors.append('* ' + tagged_string)

						if is_prob:
							for token in sentence.tokens:
								prob = token.get_tag('ner')
								prob_s = '{:<30s}{:<30s}'.format(str(token),str(prob))
								type1_errors.append(prob_s)
							type1_errors.append('\n')
			
				true_labels = []

				sentence = ""
			else:
				# Construct the sentence
				# Careful: Sentence begins with a space !
				sentence = sentence + " " + current_word 


def write_to_files():
	# Write type 1 errors to a file
	f = open('type1_errors', 'a')
	for sentence in type1_errors:
		f.write(sentence)
		f.write('\n')
	f.close()

	# Write type 2 errors to a file
	f = open('type2_errors', 'a')
	for sentence in type2_errors:
		f.write(sentence)
		f.write('\n')
	f.close()


def main():

    parser = argparse.ArgumentParser()

    best_bert_model = '/opt/kanarya/resources/flair_models/bertcustom_2020-01-09_02'
    dev_set = '/home/hasan.ozturk/kanarya-github/kanarya/data/de-da-te-ta.10E-4percent.conll.84max.dev'

    parser.add_argument("--conll_dataset_dirpath", default=dev_set)
    parser.add_argument("--flair_model_dirpath", default=best_bert_model)
    parser.add_argument("--is_prob", default=False)

    args = parser.parse_args()

    conll_dataset_dirpath = args.conll_dataset_dirpath
    flair_model_dirpath = args.flair_model_dirpath
    is_prob = args.is_prob

    params = {
    	"conll_dataset_dirpath": conll_dataset_dirpath,
    	"flair_model_dirpath": flair_model_dirpath,
    	"is_prob": is_prob
    }

    classify(params)
    write_to_files()

if __name__ == "__main__":
    main()
