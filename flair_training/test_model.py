
from flair.models import SequenceTagger
from flair.data import Sentence
import flair, torch
import sys

flair.device = torch.device('cpu')

classifier_model = sys.argv[1]
given_sentence = sys.argv[2]

classifier = SequenceTagger.load_from_file('./' + classifier_model + '/best-model.pt')

sentence = Sentence(given_sentence)

classifier.predict(sentence)
print(sentence)
#print(sentence.labels)
print(sentence.to_tagged_string())
