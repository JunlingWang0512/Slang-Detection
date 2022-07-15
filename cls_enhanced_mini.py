import random
import pandas as pd
from configuration import Configuration
from train import train_cls_enhanced
from configuration import CONSTANTS as C
from eval_test import test_enhanced_cls
import os

print(C.DEVICE)

# baseline without adapter
n_epochs_cls = 10
model_size_list = ['mini']
with_adapter_list = ['yes', 'no']
lr_cls = (6.5)*1e-7
wd_cls = 1e-3
mlm_adapter_name_list = ['model_1657682495', 'model_1657692000', 'model_1657659496']

dict_list_model = {'n_epochs_cls':[], 'lr_cls':[], 'wd_cls':[], 'model_size':[], 'test_model_dir':[], 'test_loss':[], 'mlm_adapter_name':[]}
for i in range(len(model_size_list)):
    model_size = model_size_list[i]
    mlm_adapter_name = mlm_adapter_name_list[i]
    print(model_size)
    dict_cls = {'n_epochs_cls': n_epochs_cls, 'lr_cls': lr_cls, 'wd_cls': wd_cls, 'mlm_adapter_name': mlm_adapter_name, 'model_size': model_size}
    config = Configuration(dict_cls)
    model_dir = train_cls_enhanced(config)
    dict_test = {'model_size': model_size, 'test_model_dir': model_dir}
    config = Configuration(dict_test)
    test_loss = test_enhanced_cls(config)
    for key in dict_cls.keys():
        dict_list_model[key].append(dict_cls[key])
    dict_list_model['test_model_dir'].append(model_dir)
    dict_list_model['test_loss'].append(test_loss)
    if not os.path.exists('models_result/'):
        os.makedirs('models_result/') 
    pd.DataFrame(dict_list_model).to_csv('models_result/quick_cls_summary_enhanced.csv')