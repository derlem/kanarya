#!/bin/bash
: '
# Dump all the logs to a specific file
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>/home/hasan.ozturk/outputs/log_run_pretraining_chunk1_vocab_29000_batch_16_step_1000.txt 2>&1
'
# Measure the runtime
start=`date +%s`
echo "start: $start"

python ./bert_google/run_pretraining.py \
      --input_file=$1*  \
      --output_dir=$2 \
      --do_train=True \
      --do_eval=True \
      --bert_config_file=bert_config_small_turkish.json \
      --train_batch_size=4 \
      --max_seq_length=256 \
      --max_predictions_per_seq=40 \
      --num_train_steps=1000 \
      --num_warmup_steps=10 \
      --learning_rate=2e-5


end=`date +%s`
runtime=$((end-start))

echo "end: $end"
echo "runtime: $runtime"



