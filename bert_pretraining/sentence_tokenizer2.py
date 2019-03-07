import nltk
import sys
import re

st2 = nltk.data.load('tokenizers/punkt/english.pickle')

for line in sys.stdin:
    line = re.sub(r'\s([?.!"](?:\s|$))', r'\1', line)
    sent_tokenize_list = st2.tokenize(line, realign_boundaries=True)
    for sentence in sent_tokenize_list:
        print(sentence)
        print()
