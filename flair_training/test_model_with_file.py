
from flair.models import SequenceTagger
from flair.data import Sentence
import flair, torch
import sys

flair.device = torch.device('cpu')

classifier_model = sys.argv[1]
fname = sys.argv[2]

classifier = SequenceTagger.load_from_file('./' + classifier_model + '/best-model.pt')

with open(fname) as f:
    sentences = f.readlines()
    
sentences = [sentence.strip() for sentence in sentences] 
error_number = 0
for sentence in sentences:
    sentence = Sentence(sentence)
    classifier.predict(sentence)
    print(sentence)
    if sentence.get_spans('ner'):
        error_number += 1
    #print(sentence.labels)
    print(sentence.to_tagged_string())
print('total number of sentences: ' + str(len(sentences)))
print('total number of errors found: '+ str(error_number))
