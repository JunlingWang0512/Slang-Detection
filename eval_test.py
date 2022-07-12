import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.adapters import BertAdapterModel
import pandas as pd
from configuration import CONSTANTS as C
from dataset_mlm_cls import MLMDateset, CLSDataset
from configuration import Configuration
from tqdm import tqdm
import os
import time
from train import evaluate, init_tokenizer_model

def test_baseline_cls(config):
    TEST_MODEL_DIR = 'models_cls_baseline_'+ config.model_size + '/' +config.test_model_dir +'/'
    
    tokenizer, model_cls_test = init_tokenizer_model(config)

    if config.baseline_with_adapter == 'yes':
        # model_cls_test.load_adapter(adapter_name_or_path=TEST_MODEL_DIR + 'cls_adapter/', load_as = 'cls_adapter', set_active = True)
        # model_cls_test.load_head(save_directory = TEST_MODEL_DIR + 'cls_adapter_head/', load_as = 'cls')
        model_cls_test.add_adapter('cls_adapter', set_active = True)
        model_cls_test.add_classification_head('cls')
    elif config.baseline_with_adapter == 'no':
        # model_cls_test.load_head(save_directory = TEST_MODEL_DIR + 'cls_adapter_head/', load_as = 'cls')
        model_cls_test.add_classification_head('cls')
    model_cls_test.load_state_dict(torch.load(TEST_MODEL_DIR+'state_dict.pth'))
    model_cls_test.to(C.DEVICE)

    test_cls  = pd.read_csv(C.DATA_DIR+C.TEST_CLS_CSV, index_col=0)
    testset_cls = CLSDataset(test_cls, tokenizer)
    testloader_cls = torch.utils.data.DataLoader(testset_cls, batch_size = 16, shuffle = True)

    test_loss = evaluate(model_cls_test, testloader_cls)
    print(test_loss)
    return test_loss

def test_enhanced_cls(config):
    TEST_MODEL_DIR = 'models_cls_enhanced_'+ config.model_size + '/' +config.test_model_dir +'/'
    
    tokenizer, model_cls_test = init_tokenizer_model(config)
    
    model_cls_test.add_adapter('cls_adapter', set_active = True)
    model_cls_test.add_classification_head('cls')
    model_cls_test.load_state_dict(torch.load(TEST_MODEL_DIR+'state_dict.pth'))
    model_cls_test.to(C.DEVICE)

    test_cls  = pd.read_csv(C.DATA_DIR+C.TEST_CLS_CSV, index_col=0)
    testset_cls = CLSDataset(test_cls, tokenizer)
    testloader_cls = torch.utils.data.DataLoader(testset_cls, batch_size = 16, shuffle = True)

    test_loss = evaluate(model_cls_test, testloader_cls)
    print(test_loss)
    return test_loss

if __name__ == '__main__':
    print(C.DEVICE)
    config = Configuration.parse_cmd()
    test_baseline_cls(config)