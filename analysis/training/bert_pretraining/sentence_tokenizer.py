from nltk.tokenize import sent_tokenize
import sys
import re
from time import sleep
import subprocess

for line in sys.stdin:
    line = re.sub(r'\s([?.!"](?:\s|$))', r'\1', line)
    line = re.sub('\' \'', '', line)
    line = re.sub('` `', '', line)
    line = re.sub('``', '', line)
    sent_tokenize_list = sent_tokenize(line)
    for sentence in sent_tokenize_list:
        if sentence[:-1].isdigit():
            print(sentence, end=" ")
        else:
            print(sentence)
    print()
