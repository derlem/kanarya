# Author: Hasan Öztürk
# Usage: python perturb.py --flair_model_dirpath <path-to-flair-model>  --conll_dataset_dirpath <path-to-conll-data> 
# Example: python perturb.py python perturb.py --flair_model_dirpath  /opt/kanarya/resources/flair_models/bertcustom_2020-01-09_02 --conll_dataset_dirpath /home/hasan.ozturk/kanarya-github/kanarya/data/de-da-te-ta.10E-4percent.conll.84max.dev
# Note that we only take the last word w/ the de/da suffix into account.

import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
import flair, torch
import sys
import argparse
import copy
import json

#deda_pos = 0

bert_vocab_tokens = set()
bert_vocab_subtokens = set()

delta_p_dict = {}

sentence_count = 0




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

# This will be a more convinient but costly operation, let us not it for now.
def get_deda_pos(sentence):
	deda_pos = 0
	for idx, token in enumerate(sentence.tokens):
		if does_contain_deda(token.text):
			deda_pos = idx
	return deda_pos


def populate_vocab(params):

	bert_vocab_dirpath = params["bert_vocab_dirpath"]
	global bert_vocab

	f = open(bert_vocab_dirpath, "r")
	for line in f:
		line = line.strip() # Eliminate space chars
		if line[:2] == "##":
			bert_vocab_subtokens.add(line[2:])
			##print("Subtoken: " + line[2:] + " - " + str(len(line[2:])))
		else:
			bert_vocab_tokens.add(line)
			##print("Token: " + line + " - " + str(len(line)))




def div_dict(my_dict, dividend):
	for i in my_dict:
		my_dict[i] = my_dict[i] / dividend;

# Should take an object of Sentence class, not string as an argument
def get_prob(sentence, pos):

	deda_token = sentence.tokens[pos]
	##print("Token: " + str(deda_token))
	label = deda_token.get_tag('ner') # Be careful, it returns 

	value = label.value

	if value == 'B-ERR':
		score = label.score
	else:
		score = 1 - label.score

	##print(str(deda_token) + ": " + str(label))
	return score # float


# Delete the last character of the preceding word
def perturb1(sentence, pos):

	# Preceding word
	perturbed_token_pos = max(0, pos-1)
	perturbed_sentence = copy.deepcopy(sentence)
	token = perturbed_sentence.tokens[perturbed_token_pos]
	if len(token.text) != 1:
		token.text = token.text[:-1]
	##print("Perturb 1: " + str(perturbed_sentence))
	return perturbed_sentence


# Delete the first character of the preceding word
def perturb2(sentence, pos):

	# Preceding word
	perturbed_token_pos = max(0, pos-1)
	perturbed_sentence = copy.deepcopy(sentence)
	token = perturbed_sentence.tokens[perturbed_token_pos]
	if len(token.text) != 1:
		token.text = token.text[1:]
	##print("Perturb 2: " + str(perturbed_sentence))
	return perturbed_sentence

# Capitalize the first character of the preceding word
def perturb3(sentence, pos):

	# Preceding word
	perturbed_token_pos = max(0, pos-1)
	perturbed_sentence = copy.deepcopy(sentence)
	token = perturbed_sentence.tokens[perturbed_token_pos]
	token.text = token.text.capitalize()
	##print("Perturb 3: " + str(perturbed_sentence))
	return perturbed_sentence

# !!! If there is no word in the position where a word will be deleted, then we return delta_pi = 0, which might cause problems.

# Remove the preceding word, if any
def perturb4(sentence, pos):
	
	# If there is a preceding word, then remove it. If not, do not perturb
	if pos > 0:
		perturbed_sentence = copy.deepcopy(sentence)
		del perturbed_sentence.tokens[pos - 1]
		##print("Perturb 4: " + str(perturbed_sentence))
		return perturbed_sentence
	else:
		##print("Perturb 4: " + str(sentence))
		return sentence

# Remove the two previous word, if any
def perturb5(sentence, pos):
	
	# If there is a word two before the deda_word, then remove it. If not, do not perturb
	if pos > 1:
		perturbed_sentence = copy.deepcopy(sentence)
		del perturbed_sentence.tokens[pos - 2]
		##print("Perturb 5: " + str(perturbed_sentence))
		return perturbed_sentence
	else:
		##print("Perturb 5: " + str(sentence))
		return sentence



# Remove the three previous word, if any
def perturb6(sentence, pos):
	
	# If there is a word three before the deda_word, then remove it. If not, do not perturb
	if pos > 2:
		perturbed_sentence = copy.deepcopy(sentence)
		del perturbed_sentence.tokens[pos - 3]
		##print("Perturb 6: " + str(perturbed_sentence))
		return perturbed_sentence
	else:
		##print("Perturb 6: " + str(sentence))
		return sentence

# Remove the following word, if any
def perturb7(sentence, pos):
	
	# If there is a following word, then remove it. If not, do not perturb
	if pos < (len(sentence.tokens) - 1):
		perturbed_sentence = copy.deepcopy(sentence)
		del perturbed_sentence.tokens[pos + 1]
		##print("Perturb 7: " + str(perturbed_sentence))
		return perturbed_sentence
	else:
		##print("Perturb 7: " + str(sentence))
		return sentence

# Remove the two following word, if any
def perturb8(sentence, pos):
	
	# If there is a word after 2 positions of deda_word, then remove it. If not, do not perturb
	if pos < (len(sentence.tokens) - 2):
		perturbed_sentence = copy.deepcopy(sentence)
		del perturbed_sentence.tokens[pos + 2]
		##print("Perturb 8: " + str(perturbed_sentence))
		return perturbed_sentence
	else:
		##print("Perturb 8: " + str(sentence))
		return sentence

# Remove the three following word, if any
def perturb9(sentence, pos):
	
	# If there is a word after 3 positions of deda_word, then remove it. If not, do not perturb
	if pos < (len(sentence.tokens) - 3):
		perturbed_sentence = copy.deepcopy(sentence)
		del perturbed_sentence.tokens[pos + 3]
		##print("Perturb 9: " + str(perturbed_sentence))
		return perturbed_sentence
	else:
		##print("Perturb 9: " + str(sentence))
		return sentence

# Remove the first and shortest common pattern (##subtoken) in the preceding word using BERT vocab
def perturb10(sentence, pos):

	# If there is a preceding word, then remove it. If not, do not perturb
	if pos > 0:
		perturbed_sentence = copy.deepcopy(sentence)
		
		token = perturbed_sentence.tokens[pos - 1]
		perturbed_token = perturb10_helper(token.text)

		token.text = perturbed_token
		
		
		#print("Perturb 10: " + str(perturbed_sentence)+ "\n")
		return perturbed_sentence
	else:
		#print("Perturb 10: " + str(sentence)+ "\n")
		return sentence


def perturb10_helper(word):

	# Careful w/ single chars

	for idx in range(len(word)):

		# For now ignore single chars, since it seems like there is a problem in bert vocab
		if idx > 1:
			if word[:idx+1] in bert_vocab_subtokens:
				if len(word[idx+1:]) > 0:
					#print("Perturb 10: Word: " + word + " - " + "Found: " + word[:idx+1] + " - " + "Return: " + word[idx+1:])
					return word[idx+1:]
	#print("Could not find a pattern, returning the original word: " + word)
	return word


# Keep the first and shortest common pattern (##subtoken) in the preceding word using BERT vocab and remove the rest
def perturb11(sentence, pos):

	# If there is a preceding word, then remove it. If not, do not perturb
	if pos > 0:
		perturbed_sentence = copy.deepcopy(sentence)
		
		token = perturbed_sentence.tokens[pos - 1]
		perturbed_token = perturb11_helper(token.text)

		token.text = perturbed_token
		
		
		#print("Perturb 11: " + str(perturbed_sentence)+ "\n")
		return perturbed_sentence
	else:
		#print("Perturb 11: " + str(sentence)+ "\n")
		return sentence


def perturb11_helper(word):

	# Careful w/ single chars

	for idx in range(len(word)):

		# For now ignore single chars, since it seems like there is a problem in bert vocab
		if idx > 1:
			if word[:idx+1] in bert_vocab_subtokens:
				if len(word[idx+1:]) > 0:
					#print("Perturb 11: Word: " + word + " - " + "Found: " + word[:idx+1] + " - " + "Return: " + word[:idx+1])
					return word[:idx+1]
	#print("Could not find a pattern, returning the original word: " + word)
	return word


# Remove the last and shortest common pattern (##subtoken) in the preceding word using BERT vocab
def perturb12(sentence, pos):

	# If there is a preceding word, then remove it. If not, do not perturb
	if pos > 0:
		perturbed_sentence = copy.deepcopy(sentence)
		
		token = perturbed_sentence.tokens[pos - 1]
		perturbed_token = perturb12_helper(token.text)

		token.text = perturbed_token
		
		
		#print("Perturb 12: " + str(perturbed_sentence)+ "\n")
		return perturbed_sentence
	else:
		#print("Perturb 12: " + str(sentence)+ "\n")
		return sentence


def perturb12_helper(word):

	# Careful w/ single chars

	for idx in range(len(word)):

		# For now ignore single chars, since it seems like there is a problem in bert vocab
		if idx > 1:
			##print(word[-idx:])
			if word[-(idx+1):] in bert_vocab_subtokens:
				if len(word[idx+1:]) > 0:
					#print("Perturb 12: Word: " + word + " - " + "Found: " + word[-(idx+1):] + " - " + "Return: " + word[:-(idx+1)] )
					return word[:-(idx+1):]

	#print("Could not find a pattern, returning the original word: " + word)
	return word

# Keep the last and shortest common pattern (##subtoken) in the preceding word using BERT vocab
def perturb13(sentence, pos):

	# If there is a preceding word, then remove it. If not, do not perturb
	if pos > 0:
		perturbed_sentence = copy.deepcopy(sentence)
		
		token = perturbed_sentence.tokens[pos - 1]
		perturbed_token = perturb13_helper(token.text)

		token.text = perturbed_token
		
		
		#print("Perturb 13: " + str(perturbed_sentence)+ "\n")
		return perturbed_sentence
	else:
		#print("Perturb 13: " + str(sentence)+ "\n")
		return sentence


def perturb13_helper(word):

	# Careful w/ single chars

	for idx in range(len(word)):

		# For now ignore single chars, since it seems like there is a problem in bert vocab
		if idx > 1:
			##print(word[-idx:])
			if word[-(idx+1):] in bert_vocab_subtokens:
				if len(word[idx+1:]) > 0:
					#print("Perturb 13: Word: " + word + " - " + "Found: " + word[-(idx+1):] + " - " + "Return: " + word[-(idx+1):] )
					return word[-(idx+1):]
					
	#print("Could not find a pattern, returning the original word: " + word)
	return word


# Remove the first and shortest common pattern (##subtoken) in the following word using BERT vocab
def perturb14(sentence, pos):

	# If there is a preceding word, then remove it. If not, do not perturb
	if pos < (len(sentence.tokens) - 1):
		perturbed_sentence = copy.deepcopy(sentence)
		
		token = perturbed_sentence.tokens[pos + 1]
		perturbed_token = perturb14_helper(token.text)

		token.text = perturbed_token
		
		
		#print("Perturb 14: " + str(perturbed_sentence)+ "\n")
		return perturbed_sentence
	else:
		#print("Perturb 14: " + str(sentence)+ "\n")
		return sentence


def perturb14_helper(word):

	# Careful w/ single chars

	for idx in range(len(word)):

		# For now ignore single chars, since it seems like there is a problem in bert vocab
		if idx > 1:
			if word[:idx+1] in bert_vocab_subtokens:
				if len(word[idx+1:]) > 0:
					#print("Perturb 14: Word: " + word + " - " + "Found: " + word[:idx+1] + " - " + "Return: " + word[idx+1:] )
					return word[idx+1:]
	#print("Could not find a pattern, returning the original word: " + word)
	return word

# Keep the first and shortest common pattern (##subtoken) in the following word using BERT vocab
def perturb15(sentence, pos):

	# If there is a preceding word, then remove it. If not, do not perturb
	if pos < (len(sentence.tokens) - 1):
		perturbed_sentence = copy.deepcopy(sentence)
		
		token = perturbed_sentence.tokens[pos + 1]
		perturbed_token = perturb15_helper(token.text)

		token.text = perturbed_token
		
		
		#print("Perturb 15: " + str(perturbed_sentence)+ "\n")
		return perturbed_sentence
	else:
		#print("Perturb 15: " + str(sentence)+ "\n")
		return sentence


def perturb15_helper(word):

	# Careful w/ single chars

	for idx in range(len(word)):

		# For now ignore single chars, since it seems like there is a problem in bert vocab
		if idx > 1:
			if word[:idx+1] in bert_vocab_subtokens:
				if len(word[idx+1:]) > 0:
					#print("Perturb 15: Word: " + word + " - " + "Found: " + word[:idx+1] + " - " + "Return: " + word[:idx+1])
					return word[:idx+1]
	#print("Could not find a pattern, returning the original word: " + word)
	return word

# Remove the last and shortest common pattern (##subtoken) in the following word using BERT vocab
def perturb16(sentence, pos):

	# If there is a preceding word, then remove it. If not, do not perturb
	if pos < (len(sentence.tokens) - 1):
		perturbed_sentence = copy.deepcopy(sentence)
		
		token = perturbed_sentence.tokens[pos + 1]
		perturbed_token = perturb16_helper(token.text)

		token.text = perturbed_token
		
		
		#print("Perturb 16: " + str(perturbed_sentence)+ "\n")
		return perturbed_sentence
	else:
		#print("Perturb 16: " + str(sentence)+ "\n")
		return sentence


def perturb16_helper(word):

	# Careful w/ single chars

	for idx in range(len(word)):

		# For now ignore single chars, since it seems like there is a problem in bert vocab
		if idx > 1:
			if word[-(idx+1):] in bert_vocab_subtokens:
				if len(word[idx+1:]) > 0:
					#print("Perturb 16: Word: " + word + " - " + "Found: " + word[-(idx+1):] + " - " + "Return: " + word[:-(idx+1)] )
					return word[:-(idx+1):]
	#print("Could not find a pattern, returning the original word: " + word)
	return word

# Keep the last and shortest common pattern (##subtoken) in the following word using BERT vocab
def perturb17(sentence, pos):

	# If there is a preceding word, then remove it. If not, do not perturb
	if pos < (len(sentence.tokens) - 1):
		perturbed_sentence = copy.deepcopy(sentence)
		
		token = perturbed_sentence.tokens[pos + 1]
		perturbed_token = perturb17_helper(token.text)

		token.text = perturbed_token
		
		
		#print("Perturb 17: " + str(perturbed_sentence)+ "\n")
		return perturbed_sentence
	else:
		#print("Perturb 17: " + str(sentence)+ "\n")
		return sentence


def perturb17_helper(word):

	# Careful w/ single chars

	for idx in range(len(word)):

		# For now ignore single chars, since it seems like there is a problem in bert vocab
		if idx > 1:
			if word[-(idx+1):] in bert_vocab_subtokens:
				if len(word[idx+1:]) > 0:
					#print("Perturb 17: Word: " + word + " - " + "Found: " + word[-(idx+1):] + " - " + "Return: " + word[-(idx+1):] )
					return word[-(idx+1):]
	#print("Could not find a pattern, returning the original word: " + word)
	return word




perturbation_functions = [perturb1, perturb2, perturb4, perturb5, perturb6, perturb7, perturb8, perturb9, perturb10, perturb11, perturb12, perturb13, perturb14, perturb15, perturb16, perturb17]


def create_dict():
	'''
	for func in perturbation_functions:
		delta_p_dict[str(func)] = 0
	'''

	for func in perturbation_functions:
		delta_p_dict[str(func)] = []



# !!! OMIT ZEROS

def runner(params):

	flair.device = torch.device('cpu')

	#classifier_model = sys.argv[1]
	classifier_model = params["flair_model_dirpath"]
	conll_data = params["conll_dataset_dirpath"]

	classifier = SequenceTagger.load_from_file(classifier_model + '/best-model.pt')

	true_labels = []
	f = open(conll_data, "r")

	sentence = ""
	word_idx = 0 # 0 indexing
	#deda_pos = 0
	global deda_pos
	for line in f:
		tokens = line.split()
		if(len(tokens) == 2):

			# Current word and label pair
			current_word = tokens[0] 
			current_label = tokens[1]

			'''
			# Currently we only take the last word which contains de/da in a sentence !!
			if does_contain_deda(current_word):
				deda_pos = word_idx


			word_idx += 1

			'''
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

				deda_pos = get_deda_pos(sentence)
				p0 = get_prob(sentence, deda_pos) 

				##print(sentence)
				for func in perturbation_functions:
					perturbed_sentence = func(sentence, deda_pos)

					classifier.predict(perturbed_sentence)

					deda_pos = get_deda_pos(sentence)
					pi = get_prob(perturbed_sentence, deda_pos)

					delta_p = pi - p0

					##print("Delta_p for " + str(func) + ": " + str(delta_p))
					#delta_p_dict[str(func)] += delta_p
					delta_p_dict[str(func)].append(delta_p)

				##print("=================================")
				

				sentence = ""
				word_idx = 0
				global sentence_count
				sentence_count += 1
			else:
				# Construct the sentence
				# Careful: Sentence begins with a space !
				sentence = sentence + " " + current_word


def report():

	#global delta_p_dict

    print("Sentence Count: " + str(sentence_count) + "\n")

    print('{:<50s}{:<30s}{:<30s}{:<30s}'.format("Function","Absolute Mean", "Direction", "Variance"))
    for func in perturbation_functions:
    	# Eliminate zeros (?)
    	delta_p_dict[str(func)] = [i for i in delta_p_dict[str(func)] if i != 0.0]

    	mean = np.mean(delta_p_dict[str(func)])
    	var = np.var(delta_p_dict[str(func)])
    	direction = "+" if mean > 0 else "-"

    	s = '{:<50s}{:<30s}{:<30s}{:<30s}'.format(str(func),str(abs(mean)), direction, str(var))
    	print(s)



def main():

    parser = argparse.ArgumentParser()

    # Default parameter values
    best_bert_model = '/opt/kanarya/resources/flair_models/bertcustom_2020-01-09_02'
    test_set = '/home/hasan.ozturk/kanarya-github/kanarya/data/de-da-te-ta.10E-4percent.conll.84max.test'
    bert_vocab_dirpath = '/opt/kanarya/resources/vocabs/vocab_whole_corpus_28996.txt'

    parser.add_argument("--conll_dataset_dirpath", default=test_set)
    parser.add_argument("--flair_model_dirpath", default=best_bert_model)
    parser.add_argument("--bert_vocab_dirpath", default=bert_vocab_dirpath)

    args = parser.parse_args()

    conll_dataset_dirpath = args.conll_dataset_dirpath
    flair_model_dirpath = args.flair_model_dirpath

    params = {
    	"conll_dataset_dirpath": conll_dataset_dirpath,
    	"flair_model_dirpath": flair_model_dirpath,
    	"bert_vocab_dirpath": bert_vocab_dirpath
    }

    populate_vocab(params)

    create_dict()
    runner(params)

    report()

    '''
    global delta_p_dict

    #print("Sentence Count: " + str(sentence_count))

    div_dict(delta_p_dict, sentence_count)
    # Pretty #print
    #print (json.dumps(delta_p_dict, indent=4))
    '''


if __name__ == "__main__":
    main()

