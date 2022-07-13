import random
import pandas as pd
from configuration import Configuration
from train import train_cls_baseline
from configuration import CONSTANTS as C
from eval_test import test_baseline_cls
import os

print(C.DEVICE)

# baseline without adapter
n_epochs_cls = 10
model_size_list = [ 'large', 'base', 'mini']
with_adapter_list = ['yes', 'no']
lr_cls = 1e-6
wd_cls = 1e-3

dict_list_model = {'baseline_with_adapter':[], 'n_epochs_cls':[], 'lr_cls':[], 'wd_cls':[], 'model_size':[], 'test_model_dir':[], 'test_loss':[]}
for model_size in model_size_list:
    for with_adapter in with_adapter_list:
        print(model_size)
        dict_cls = {'baseline_with_adapter': with_adapter, 'n_epochs_cls': n_epochs_cls, 'lr_cls': lr_cls, 'wd_cls': wd_cls, 'model_size': model_size}
        config = Configuration(dict_cls)
        model_dir = train_cls_baseline(config)
        dict_test = {'model_size': model_size, 'test_model_dir': model_dir, 'baseline_with_adapter': with_adapter}
        config = Configuration(dict_test)
        test_loss = test_baseline_cls(config)
        for key in dict_cls.keys():
            dict_list_model[key].append(dict_cls[key])
        dict_list_model['test_model_dir'].append(model_dir)
        dict_list_model['test_loss'].append(test_loss)
        if not os.path.exists('models_result/'):
            os.makedirs('models_result/') 
        pd.DataFrame(dict_list_model).to_csv('models_result/quick_cls_summary_baseline.csv')