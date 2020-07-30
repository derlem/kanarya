# Step by Step Guide for BERT Pretraining


First of all, these [requirements](https://github.com/derlem/kanarya/blob/bert_pretraining/requirements.txt) should be installed.

There are multiple steps required to create BERT embeddings with custom data. Firstly, BERT requires a Wordpiece vocabulary. However,
Wordpiece library is not open source. Therefore we use a different library developed by Google called [SentencePiece](https://github.com/google/sentencepiece) 
which follows a similar strategy with Wordpiece. However, this requires a conversion since these two libraries follow a different format.

After creating the vocabulary, a tensorflow record should be created using the custom data and the vocabulary. This is required since BERT
pretraining script requires a tfrecord to process.

Last and the most time consuming step is to run pretraining. Hyperparameters should be selected wisely.

## Create a Vocabulary

- Navigate to bert_pretraining directory.
- Run the `run_sentencepiece_trainer.sh` script with the corpus(one line per sentence). An example would be: `./run_sentencepiece_trainer.sh /opt/kanarya/corpus/turkish-texts-tokenized.txt`
This will create 2 files called "m.model" and "m.vocab" in the current directory.
- Next step is to make a conversion in the vocab format. To do this: `python vocab_converter.py`
- Do not forget to adjust the vocabulary size inside the `vocab_converter.py` and `sentencepiece_trainer.py`. 
 
## Create Pretraining Data

- As described above, before pretraining we should create a pretraining data. This pretraining data is a file with an extension of tfrecord
and it is created with the corpus and the vocabulary. Here is an example:
- `./run_create_pretraining_data.sh /opt/kanarya/corpus/chunks/chunk_1.txt /home/hasan.ozturk/outputs/chunk1.tfrecord /home/hasan.ozturk/inputs/vocab.txt`
- Note that this operation requires memory proportional to the size of the input file and running with the whole corpus(~6 GB) does not 
fit the RAM. Therefore we process the whole corpus chunk by chunk:
- `run_create_pretraining_data_chunks.sh`

## Run pretraining 

- After everything is ready, we run the pretraining with the tfrecord created in the previous step. This will yield an output which 
is planned to use as BERT embeddings in the flair training. An example:
- `./run_pretraining.sh /home/hasan.ozturk/outputs/chunk1.tfrecord /home/hasan.ozturk/outputs/chunk1_pretraining_output`


# Tips

- If there is job started to run on GPU and suspended (Ctrl + Z) after a while, it still uses GPU memory even if is suspended. Therefore
trying to run another job which requires GPU will most probably give an error of "CUDA out of memory". Therefore suspended jobs should be 
killed, if there is a need to run another GPU intensive job.
- If there is an "out of memory" error, try reducing the batch size.




