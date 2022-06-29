import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.adapters import BertAdapterModel
import pandas as pd
from configuration import CONSTANTS as C
from dataset_mlm_cls import MLMDateset, CLSDataset
from configuration import Configuration
import os
import time


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.padding_side = "left" 
tokenizer.pad_token = '[PAD]' # to avoid an error
tokenizer.mask_token = '[MASK]'


def adapter_not_update_cls(model_cls):
    for name, para in model_cls.named_parameters():
        if 'adapter' in name:
            para.requires_grad = False

def evaluate(model, dataloader):
    model.eval()
    loss_agg = 0
    cnt = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss_agg += loss.cpu().item() * len(batch)
            cnt += len(batch)
    
    total_loss = loss_agg/cnt
    return total_loss


def train_mlm(config):
    model_mlm = BertAdapterModel.from_pretrained('bert-base-uncased')
    model_mlm.add_adapter('mlm_adapter', set_active = True)
    model_mlm.add_masked_lm_head('mlm_head')
    model_mlm.to(C.DEVICE)

    train_mlm = list(pd.read_csv(C.DATA_DIR+config.mlm_train_name)['generate'])[:100]
    eval_mlm = list(pd.read_csv(C.DATA_DIR+config.mlm_eval_name)['generate'])[:100]

    trainset_mlm = MLMDateset(train_mlm, tokenizer)
    evalset_mlm = MLMDateset(eval_mlm, tokenizer)

    trainloader_mlm = torch.utils.data.DataLoader(trainset_mlm, batch_size = 16, shuffle = True)
    evalloader_mlm = torch.utils.data.DataLoader(evalset_mlm, batch_size = 16, shuffle = True)

    optimizer_mlm = AdamW(model_mlm.parameters(), lr = 1e-5)

    best_val_loss = 10000
    for epoch in range(config.n_epochs_mlm):
        loss_agg = 0
        cnt = 0
        start = time.time()
        for i, batch in enumerate(trainloader_mlm):
            optimizer_mlm.zero_grad()
            outputs = model_mlm(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer_mlm.step()

            loss_agg += loss.cpu().item() * len(batch)
            cnt += len(batch)

        train_loss = loss_agg/cnt
        end = time.time()
        elapsed = end-start
        print('[TRAIN MLM {:0>4d} / {:0>4d}] loss: {:.6f}, elapsed {:.3f} secs'.format(epoch, config.n_epochs_mlm, train_loss, elapsed))

        start = time.time()
        eval_loss = evaluate(model_mlm, evalloader_mlm)
        end = time.time()
        elapsed = end-start
        print('[VALID MLM {:0>4d} / {:0>4d}] loss: {:.6f}, elapsed {:.3f} secs'.format(epoch, config.n_epochs_mlm, eval_loss, elapsed))

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            if not os.path.exists(C.MODEL_DIR+'mlm_adapter/'):
                os.makedirs(C.MODEL_DIR+'mlm_adapter/') 
            model_mlm.save_adapter(save_directory=C.MODEL_DIR + 'mlm_adapter/', adapter_name = 'mlm_adapter', with_head = False)
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict':optimizer_mlm.state_dict(),
                'train_loss': train_loss,
                'eval_loss': eval_loss
            }, C.MODEL_DIR+'model_mlm.pth')



def train_cls(config):
    model_cls = BertAdapterModel.from_pretrained('bert-base-uncased')
    if config.enhance == True:
        model_cls.load_adapter(adapter_name_or_path=C.MODEL_DIR + 'mlm_adapter/', load_as = 'cls_adapter', set_active = True)
    else:
        model_cls.load_adapter(adapter_name_or_path=C.MODEL_DIR + 'mlm_adapter/', load_as = 'cls_adapter', set_active = False)
        
    model_cls.add_classification_head('cls')
    model_cls.to(C.DEVICE)

    if config.update_adapter_cls == False:
        adapter_not_update_cls(model_cls)

    train_cls  = pd.read_csv(C.DATA_DIR+C.TRAIN_CLS, index_col=0)[:100]
    eval_cls  = pd.read_csv(C.DATA_DIR+C.EVAL_CLS, index_col=0)[:100]

    trainset_cls = CLSDataset(train_cls, tokenizer)
    evalset_cls = CLSDataset(eval_cls, tokenizer)

    trainloader_cls = torch.utils.data.DataLoader(trainset_cls, batch_size = 16, shuffle = True)
    evalloader_cls = torch.utils.data.DataLoader(evalset_cls, batch_size = 16, shuffle = True)

    optimizer_cls = AdamW(model_cls.parameters(), lr = 1e-5)

    best_val_loss = 10000
    for epoch in range(config.n_epochs_cls):
        loss_agg = 0
        cnt = 0
        start = time.time()
        for i, batch in enumerate(trainloader_cls):

            optimizer_cls.zero_grad()
            outputs = model_cls(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer_cls.step()

            loss_agg += loss.cpu().item() * len(batch)
            cnt += len(batch)

        train_loss = loss_agg/cnt
        end = time.time()
        elapsed = end-start

        print('[TRAIN CLS {:0>5d} / {:0>3d}] loss: {:.6f}, elapsed {:.3f} secs'.format(epoch, config.n_epochs_mlm, train_loss, elapsed))

        start = time.time()
        eval_loss = evaluate(model_cls, evalloader_cls)
        end = time.time()
        elapsed = end-start
        print('[VALID CLS {:0>5d} / {:0>3d}] loss: {:.6f}, elapsed {:.3f} secs'.format(epoch, config.n_epochs_mlm, eval_loss, elapsed))

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            if config.enhance == True and config.update_adapter_cls == True:
                if not os.path.exists(C.MODEL_DIR + 'cls_adapter/'):
                    os.makedirs(C.MODEL_DIR + 'cls_adapter/') 
                model_cls.save_adapter(save_directory=C.MODEL_DIR + 'cls_adapter/', adapter_name = 'cls_adapter', with_head = True)
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict':optimizer_cls.state_dict(),
                'train_loss': train_loss,
                'eval_loss': eval_loss
            }, C.MODEL_DIR+'model_cls.pth')


if __name__ == '__main__':
    config = Configuration.parse_cmd()
    train_mlm(config)
    train_cls(config)