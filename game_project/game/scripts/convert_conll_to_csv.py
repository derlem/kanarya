#import numpy as np
#from flair.models import SequenceTagger
#from flair.data import Sentence
#import flair, torch
#sys
import argparse
#import copy
#import json

def does_contain_deda(word):
    suffixes = ['de','da','te','ta']
    if word[-2:] in suffixes:
        return True
    else:
        return False

# Returns position of first deda
def get_deda_pos(sentence):

    tokens = sentence.split()
    for idx, token in enumerate(tokens):
        if does_contain_deda(token):
            return idx

def write_to_csv(params, idx, sentence, pos):

    destination_dirpath = params['destination_dirpath']
    destination_dirpath += 'sentences.csv'
    f = open(destination_dirpath, 'a')
    line = f"{idx},\"{sentence}\",{pos}\n"
    f.write(line)
    f.close()

def runner(params):

    #flair.device = torch.device('cpu')
    conll_data = params["conll_dataset_dirpath"]
    f = open(conll_data, "r")

    sentence_idx = 0
    sentence = ""
    does_contain_error = False
    for line in f:
        tokens = line.split()

        # if it is the end of a sentence
        if(line == '\n'):

            # Take the sentences which does not have any error
            if not does_contain_error:
                pos = get_deda_pos(sentence)
                write_to_csv(params, sentence_idx, sentence, pos)
                sentence_idx += 1
            sentence = ""
            does_contain_error = False
        else:
            current_word = tokens[0]
            current_label = tokens[1]
            if(current_label == 'B-ERR'):
                does_contain_error = True

            if sentence == "":
                sentence = current_word
            else:
                sentence = sentence + " " + current_word





def main():

    parser = argparse.ArgumentParser()

    train_set = '/home/tony/Desktop/491/kanarya/game_project/game/static/game/de-da-te-ta.10E-3percent.conll.84max.train'
    path_to_static_folder = '/home/tony/Desktop/491/kanarya/game_project/game/static/game/'

    parser.add_argument("--conll_dataset_dirpath", default=train_set)
    parser.add_argument("--destination_dirpath", default=path_to_static_folder)

    args = parser.parse_args()

    conll_dataset_dirpath = args.conll_dataset_dirpath
    destination_dirpath = args.destination_dirpath

    params = {
    	"conll_dataset_dirpath": conll_dataset_dirpath,
        "destination_dirpath": destination_dirpath
    }

    runner(params)


if __name__ == "__main__":
    main()