from nltk.tokenize import sent_tokenize

import sys

for line in sys.stdin:
    sent_tokenize_list = sent_tokenize(line)
    for sentence in sent_tokenize_list:
        print(sentence)
        print()