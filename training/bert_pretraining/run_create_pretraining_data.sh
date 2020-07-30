#!/bin/bash

# Dump all the logs to a specific file
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>/home/hasan.ozturk/outputs/log.txt 2>&1

# Measure the runtime
start=`date +%s`
echo "start: $start"

python ./bert_new/bert/create_pretraining_data.py \
        --input_file=$1 \
        --output_file=$2 \
	--vocab_file=$3 \
	--do_lower_case=False \
	--max_seq_length=256 \
	--max_predictions_per_seq=40 \
	--masked_lm_prob=0.15 \
	--random_seed=12345 \
	--dupe_factor=5

end=`date +%s`
runtime=$((end-start))

echo "end: $end"
echo "runtime: $runtime"
