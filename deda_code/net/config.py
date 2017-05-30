from datetime import datetime
import os


in_dim = 300
out_dim = 1
save_fol = '/Users/miracgoksuozturk/Documents/workspace_python/deda_code/models'
ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
save_file_path = os.path.join(save_fol, 'Model_%s' % ts)
reload_model = False
reload_model_path = '/Users/miracgoksuozturk/Documents/workspace_python/deda_code/models/Model_2017-05-23-16-34'
hidden_layer_size = [256, 256, 256, 128]
hidden_layer_type = ['TANH', 'TANH', 'TANH', 'LSTM']
batch_size = 1
lr = 0.00005
half_lr = False
warmup_epoch = 1
epoch = 500
drp_rate = 0.5
patience = 10
momentum = 0.9
warmup_momentum = 0.3
opt = 'adam'
save_model = True
trust_input = True
