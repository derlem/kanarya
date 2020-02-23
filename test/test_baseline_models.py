import argparse
import os

files = [os.path.join("../data", filename) for filename in ['de-da-te-ta.10E-4percent.conll.dev',
                                                           'de-da-te-ta.10E-4percent.conll.train',
                                                           'de-da-te-ta.10E-4percent.conll.test',
                                                           'de-da-te-ta.10E-4percent.conll.84max.dev',
                                                           'de-da-te-ta.10E-4percent.conll.84max.train',
                                                           'de-da-te-ta.10E-4percent.conll.84max.test']]


def read_corpus(files):
    corpus = ''
    for file in files:
        with open(file, 'r') as f:
            corpus += '\n' + f.read()
    return corpus


def calculate_metrics(corpus, baseline_model_type):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    corpus = corpus.split('\n\n')
    for sentence in corpus:
        pairs = sentence.split('\n')
        for pair in pairs:
            if pair == '':
                continue
            word = pair.split()[0]
            tag = pair.split()[1]
            if word == 'de' or word == 'da':
                if tag == 'B-ERR':
                    if baseline_model_type == "disjoint":
                        fn += 1
                    else:
                        tp += 1
                elif tag == 'O':
                    if baseline_model_type == "disjoint":
                        tn += 1
                    else:
                        fp += 1
            elif word[-2:] == 'de' or word[-2:] == 'da' or word[-2:] == 'te' or word[-2:] == 'ta':
                if tag == 'B-ERR':
                    if baseline_model_type == "disjoint":
                        tp += 1
                    else:
                        fn += 1
                elif tag == 'O':
                    if baseline_model_type == "disjoint":
                        fp += 1
                    else:
                        tn += 1
    return tp, tn, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_type", choices=["joint", "disjoint"], required=True)

    args = parser.parse_args()

    baseline_model_type = args.baseline_model_type

    assert baseline_model_type in ["joint", "disjoint"], "only "

    corpus = read_corpus(files)
    tp, tn, fp, fn = calculate_metrics(corpus, baseline_model_type)

    print('true positive ' + str(tp))
    print('true negative ' + str(tn))
    print('false positive ' + str(fp))
    print('false negative ' + str(fn))


if __name__ == "__main__":
    main()
