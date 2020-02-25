# Author: Hasan Öztürk
# Usage: python extract_errors.py <path-to-flair-model> <path-to-conll-data>
# Example: python extract_errors.py /opt/kanarya/resources/flair_models/bertcustom_2020-01-09_02 /home/hasan.ozturk/kanarya-github/kanarya/data/de-da-te-ta.10E-4percent.conll.84max.dev

import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
import flair, torch
import sys

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

flair.device = torch.device('cpu')

classifier_model = sys.argv[1]
fname = sys.argv[2]

classifier = SequenceTagger.load_from_file(classifier_model + '/best-model.pt')

type1_errors = []
type2_errors = []

true_labels = []
f = open(fname, "r")

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
					type2_errors.append(tagged_string)
				# The model find an error when there is not (False positive)
				else:
					type1_errors.append(tagged_string)
			
			true_labels = []

			sentence = ""
		else:
			# Construct the sentence
			# Careful: Sentence begins with a space !
			sentence = sentence + " " + current_word 

# Write type 1 errors to a file
for sentence in type1_errors:
	f = open('type1_errors', 'a')
	f.write(sentence)
	f.write('\n')
	f.close()

# Write type 2 errors to a file
for sentence in type2_errors:
	f = open('type2_errors', 'a')
	f.write(sentence)
	f.write('\n')
	f.close()
