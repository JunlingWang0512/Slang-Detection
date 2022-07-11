import random
from configuration import Configuration
from train import train_mlm, train_cls

# mlm search
lr_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
for i in range(10):
    mlm_threshold = random.randint(0,10) * 0.1
    n_epochs_mlm = random.randint(1,10)
    lr_mlm = lr_list[random.randint(1, len(lr_list))]
    dict_mlm = {'mlm_threshold': mlm_threshold, 'n_epochs_mlm': n_epochs_mlm, 'lr_mlm': lr_mlm}
    config = Configuration(dict_mlm)
    train_mlm(config)


# cls search +mlm, with adapter update
for i in range(10):
    n_epochs_cls = random.randint(1,10)
    lr_cls = lr_list[random.randint(1, len(lr_list))]
    dict_cls = {'baseline': False, 'mlm_adapter_name': 'model_1657482929', 'n_epochs_cls': n_epochs_cls,'lr_cls': lr_cls, 'update_adapter_cls': True}
    config = Configuration(dict_cls)
    train_cls(config)