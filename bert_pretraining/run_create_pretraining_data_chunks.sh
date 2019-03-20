#!/bin/bash
input_file="../sentence_split_chunk"
vocab_file="../m.model"
output_file="pretraining_datas/data"

../sentence_split_chunk1.txt pretraining_datas/data1.tfrecord ../m.model 
for i in {1..12}
do
	input_file="$input_file$i.txt"
	output_file="$output_file$i.tfrecord"
	echo "Started creating pretraining data for file: $input_file"
	./run_create_pretraining_data.sh $input_file $output_file $vocab_file
	echo "Completed creating pretraining data for file: $input_file"
	input_file="../sentence_split_chunk"
	output_file="pretraining_datas/data"
done