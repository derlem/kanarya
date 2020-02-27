# Use the BERT model of interest and update the following path
export BERT_BASE_DIR=/home/hasan.ozturk/outputs/bert_models/whole_vocab_28996_lr_2e-5_step_290000_seq_256_pretraining_output

: '
  In order to give tensorflow BERT model as an argument:
  - Rename model.ckpt-<checkpoint_num>.data* file to model.ckpt-<checkpoint_num>
  Basically use the longest common prefix among the tf BERT output files
'
transformers bert \
  $BERT_BASE_DIR/model.ckpt-290000 \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
