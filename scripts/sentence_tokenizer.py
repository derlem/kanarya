from nltk.tokenize import sent_tokenize

import sys

for line in sys.stdin:
    # print line
    sent_tokenize_list = sent_tokenize(line.decode("utf8"))
    for sentence in sent_tokenize_list:
        print sentence.encode("utf8")