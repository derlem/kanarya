
import sentencepiece as spm
import argparse

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_sentence_number(fname):
    number_of_sentences = 0

    with open() as f:
        for line in f:
           if line.strip():
              number_of_sentences += 1
    return number_of_sentences

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--model_prefix", default="m")
    parser.add_argument("--vocab_size", type=int, default=28996)
    parser.add_argument("--model_type", default="bpe")
    parser.add_argument("--input_sentence_size", type=int)
    parser.add_argument("--control_symbols", default="[SEP],[CLS],[PAD],[MASK]")

    args = parser.parse_args()

    sentence_size = file_len(args.input)

    spm.SentencePieceTrainer.Train \
        ("--input=%s "
         "--model_prefix=%s "
         "--vocab_size=%d "
         "--character_coverage=1.0 "
         "--model_type=%s "
         "--input_sentence_size=%d "
         "--control_symbols=%s" %
        (args.input, args.model_prefix, int(args.vocab_size), args.model_type, sentence_size, args.control_symbols))
