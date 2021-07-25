#!/bin/bash

BERT_MODEL_DIR=$1

transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_MODEL_DIR/bert_model.ckpt \
  --config $BERT_MODEL_DIR/bert_config.json \
  --pytorch_dump_output $BERT_MODEL_DIR/pytorch_model.bin