#!/bin/bash
input_file="../../corpus/chunks/chunk_"
vocab_file="m.model"
output_file="./bert/pretraining_datas/pretraining_datas_seq_num_256/data_"
 
for i in {1..51}
do
    input_file="$input_file$i.txt"
    output_file="$output_file$i.tfrecord"
    echo "Started creating pretraining data for file: $input_file"
    ./run_create_pretraining_data.sh $input_file $output_file $vocab_file
    echo "Completed creating pretraining data for file: $output_file"
    input_file="../../corpus/chunks/chunk_"
    output_file="./bert/pretraining_datas/pretraining_datas_seq_num_256/data_"
done
