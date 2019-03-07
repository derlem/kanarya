from nltk.tokenize import sent_tokenize

import sys

import re

for line in sys.stdin:
    line = re.sub(r'\s([?.!"](?:\s|$))', r'\1', line)
    sent_tokenize_list = sent_tokenize(line)
    for sentence in sent_tokenize_list:
        print(sentence)
        print()