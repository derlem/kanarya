
from flair.models import SequenceTagger
from flair.data import Sentence
import flair, torch
import sys

flair.device = torch.device('cpu')

classifier_model = sys.argv[1]

classifier = SequenceTagger.load_from_file('./' + classifier_model + '/best-model.pt')

print('enter the sentence:')
sentence_input = input()

sentence = Sentence(sentence_input)

classifier.predict(sentence)
#print(sentence)
#print(sentence.labels)
print(sentence.to_tagged_string())
while sentence is not 'EXIT':
    print('enter the sentence:')
    sentence_input = input()
    sentence = Sentence(sentence_input)
    classifier.predict(sentence)
    print(sentence.to_tagged_string())


