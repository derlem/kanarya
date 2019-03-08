
import sentencepiece as spm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--model_prefix", default="m")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--model_type", default="unigram")
    #parser.add_argument("--input_sentence_size", type=int)
    parser.add_argument("--control_symbols", default="[SEP],[CLS],[PAD],[MASK]")

    args = parser.parse_args()

    spm.SentencePieceTrainer.Train \
        ("--input=%s "
         "--model_prefix=%s "
         "--vocab_size=%d "
         "--character_coverage=1.0 "
         "--model_type=%s "
         #"--input_sentence_size=%d "
         "--control_symbols=%s" %
        (args.input, args.model_prefix, int(args.vocab_size), args.model_type, args.control_symbols))