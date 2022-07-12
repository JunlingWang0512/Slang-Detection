import random
import pandas as pd
from fontTools import configLogger
from configuration import Configuration
from train import train_cls_enhanced
from configuration import CONSTANTS as C
from eval_test import test_enhanced_cls
import os

print(C.DEVICE)

# baseline without adapter
n_epochs_cls = 2
model_size_list = ['mini', 'base', 'large']
with_adapter_list = ['yes', 'no']
lr_list = [1e-7, 5e-7, 1e-6, 1e-5]
wd_list = [1e-4, 1e-3, 1e-2, 5e-2] 
model_size = 'mini'
lr_cls = 1e-6
wd_cls = 1e-2


# enhanced model
dict_cls = {'n_epochs_cls': n_epochs_cls, 'lr_cls': lr_cls, 'wd_cls': wd_cls, 'mlm_adapter_name': 'model_1657588029', 'model_size': model_size}
config = Configuration(dict_cls)
model_dir = train_cls_enhanced(config)
dict_test = {'model_size': model_size, 'test_model_dir': model_dir}
config = Configuration(dict_test)
test_loss = test_enhanced_cls(config)