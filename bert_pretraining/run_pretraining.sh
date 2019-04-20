echo "started pretraining at $(date)"
python3 ./bert/run_pretraining.py --input_file=$1*  --output_dir=$2 --do_train=True --do_eval=True --bert_config_file=./bert/bert_config-small-turkish.json --train_batch_size=56 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=26500000 --num_warmup_steps=10 --learning_rate=2e-5
echo "ended pretraining at $(date)"
