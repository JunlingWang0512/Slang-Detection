from configuration import Configuration
from train import train_mlm

# mlm search

n_epochs_mlm = 10
model_size_list = ['large', 'base', 'mini']
lr_mlm = 1e-5
wd_mlm = 1e-2

for model_size in model_size_list:
    print(model_size)
    # try:
    mlm_threshold = 0.5
    dict_mlm = {'mlm_threshold': mlm_threshold, 'n_epochs_mlm': n_epochs_mlm, 'lr_mlm': lr_mlm, 'wd_mlm': wd_mlm, 'model_size': model_size}
    config = Configuration(dict_mlm)
    train_mlm(config)