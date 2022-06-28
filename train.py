import torch
from torch.optim import AdamW
from transformers import BertTokenizer
from transformers.adapters import BertAdapterModel
import pandas as pd
from configuration import CONSTANTS as C
from dataset_mlm_cls import MLMDateset, CLSDataset
from configuration import Configuration
import os


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.padding_side = "left" 
tokenizer.pad_token = '[PAD]' # to avoid an error
tokenizer.mask_token = '[MASK]'


def adapter_not_update_cls(model_cls):
    for name, para in model_cls.named_parameters():
        if 'adapter' in name:
            para.requires_grad = False


def train_mlm(config):
    model_mlm = BertAdapterModel.from_pretrained('bert-base-uncased')
    model_mlm.add_adapter('mlm_adapter', set_active = True)
    model_mlm.add_masked_lm_head('mlm_head')
    model_mlm.to(C.DEVICE)

    data_mlm = list(pd.read_csv(C.DATA_DIR+config.mlm_name)['generate'])[:100]
    dataset_mlm = MLMDateset(data_mlm, tokenizer)
    trainloader_mlm = torch.utils.data.DataLoader(dataset_mlm, batch_size = 16, shuffle = True)
    optimizer = AdamW(model_mlm.parameters(), lr = 1e-5)
    for epoch in range(config.n_epochs_mlm):
        for i, batch in enumerate(trainloader_mlm):
            print(i)
            optimizer.zero_grad()
            outputs = model_mlm(**batch)
            loss = outputs.loss
            print(loss)
            loss.backward()
            optimizer.step()

    if not os.path.exists(C.ADAPTER_DIR):
        os.makedirs(C.ADAPTER_DIR)
        print("save adapter at directory " , C.ADAPTER_DIR ,  " Created ")
    else:    
        print("save adapter at directory " , C.ADAPTER_DIR ,  " already exists")    
    model_mlm.save_adapter(save_directory=C.ADAPTER_DIR + 'ad_' + str(C.TIME) + '/', adapter_name = 'mlm_adapter', with_head = True)



def train_cls(config):
    model_cls = BertAdapterModel.from_pretrained('bert-base-uncased')

    if config.activate_adapter_cls == True:
        model_cls.load_adapter(adapter_name_or_path=C.ADAPTER_DIR + 'ad_' + str(C.TIME) + '/', set_active = True)

    model_cls.add_classification_head('cls')
    model_cls.to(C.DEVICE)

    if config.update_adapter_cls == False:
        adapter_not_update_cls(model_cls)

    data_train_cls  = pd.read_csv(C.DATA_DIR+C.TRAIN_CLS, index_col=0)[:100]
    trainset_cls = CLSDataset(data_train_cls, tokenizer)
    trainloader_cls = torch.utils.data.DataLoader(trainset_cls, batch_size = 16, shuffle = True)
    optimizer_cls = AdamW(model_cls.parameters(), lr = 1e-5)

    for epoch in range(config.n_epochs_cls):
        for i, batch in enumerate(trainloader_cls):
            print(i)
            optimizer_cls.zero_grad()
            outputs = model_cls(**batch)
            loss = outputs.loss
            print(loss)
            loss.backward()
            optimizer_cls.step()

if __name__ == '__main__':
    config = Configuration.parse_cmd()
    train_mlm(config)
    train_cls(config)