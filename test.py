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
from train import evaluate

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.padding_side = "left" 
tokenizer.pad_token = '[PAD]' # to avoid an error
tokenizer.mask_token = '[MASK]'

def test_cls(config):
    model_cls_test = BertAdapterModel.from_pretrained('bert-base-uncased')
    if config.baseline == False or config.baseline_with_adapter == True:
        model_cls_test.load_adapter(adapter_name_or_path=C.MODEL_DIR + 'cls_adapter/', load_as = 'cls_adapter', set_active = True)
    else:
        model_cls_test.load_adapter(adapter_name_or_path=C.MODEL_DIR + 'cls_adapter/', load_as = 'cls_adapter', set_active = False)
    model_cls_test.to(C.DEVICE)

    test_cls  = pd.read_csv(C.DATA_DIR+C.TEST_CLS_CSV, index_col=0)
    testset_cls = CLSDataset(test_cls, tokenizer)
    testloader_cls = torch.utils.data.DataLoader(testset_cls, batch_size = 16, shuffle = True)

    test_loss = evaluate(model_cls_test, testloader_cls)
    
    return test_loss