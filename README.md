# kanarya
kanarya aims to correct writing errors of the Turkish conjunction de/da by making a semantical analysis with machine learning and natural language processing.

## Description

In Turkish, the conjunction de/da should always be written seperately and never attached to any word. It is usually written mistakenly attached to a word and confused with the location suffix de/da that is always attached to a word due to them being phonetically exactly same and to lack of grammatical rule to distinguish these two. In fact the only way to determine the correct writing of de/da is to sementically determining its role in the sentence. Some basic information on this subject can be further found [here](https://fluentinturkish.com/grammar/conjunctions).

kanarya aims to solve this problem with pre-training of BERT model with the Turkish corpus. You can learn more about BERT with its [paper](https://arxiv.org/abs/1810.04805) and check its [github repo](https://github.com/google-research/bert)
