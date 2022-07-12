import torch
from torch.optim import AdamW
from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, BertTokenizer, BertForSequenceClassification
from transformers.adapters import BertAdapterModel
import pandas as pd
from configuration import CONSTANTS as C
from dataset_mlm_cls import MLMDateset, CLSDataset
from configuration import Configuration
from tqdm import tqdm
import os
import time

def print_bert_para(model_cls):
    for name, para in model_cls.named_parameters():
        print(name, para)

def mlm_freeze_bert(model_cls):
    for name, para in model_cls.named_parameters():
        if not ('mlm_adapter' in name or 'mlm_head' in name):
            para.requireds_grad = False

def adapter_not_update_cls(model_cls):
    for name, para in model_cls.named_parameters():
        # print('require grad', para.required_grad)
        if 'adapter' in name:
            para.requires_grad = False

def init_tokenizer_model(config):
    if config.model_size == 'mini':
        model_name = 'prajjwal1/bert-mini'
    elif config.model_size == 'base':
        model_name = 'bert-base-uncased'
    elif config.model_size == 'large':
        model_name = 'bert-large-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = '[PAD]' # to avoid an error
    tokenizer.mask_token = '[MASK]'

    model = BertAdapterModel.from_pretrained(model_name)

    return tokenizer, model
    

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
    tokenizer, model_mlm = init_tokenizer_model(config)

    model_mlm.add_adapter('mlm_adapter', set_active = True)
    model_mlm.add_masked_lm_head('mlm_head')
    model_mlm.to(C.DEVICE)
    mlm_freeze_bert(model_mlm)

    print('load training data')
    train_mlm = pd.read_csv(C.DATA_DIR+C.TRAIN_MLM_CSV)
    trainset_mlm = MLMDateset(train_mlm, tokenizer, config)
    trainloader_mlm = torch.utils.data.DataLoader(trainset_mlm, batch_size = 16, shuffle = True)

    print('load evaluating data')
    eval_mlm = pd.read_csv(C.DATA_DIR+C.EVAL_MLM_CSV)
    evalset_mlm = MLMDateset(eval_mlm, tokenizer, config)
    evalloader_mlm = torch.utils.data.DataLoader(evalset_mlm, batch_size = 16, shuffle = True)

    print('data loaded')

    optimizer_mlm = AdamW(model_mlm.parameters(), lr = config.lr_mlm, weight_decay = config.wd_mlm)

    best_val_loss = 10000
    glob_cnt = 0

    MODEL_MLM_DIR = 'models_mlm_' + config.model_size + '/model_'+str(int(time.time())) + '/'
    if not os.path.exists(MODEL_MLM_DIR+'mlm_adapter/'):
        os.makedirs(MODEL_MLM_DIR+'mlm_adapter/') 

    config.to_json(MODEL_MLM_DIR + 'config.json')

    dict_record = {'batch': [], 'epoch':[], 'glob_cnt':[], 'train_loss': [], 'valid_loss': []}
    start = time.time()
    for epoch in range(config.n_epochs_mlm):
        epoch_start = time.time()
        for i, batch in enumerate(trainloader_mlm):
            # print(i)
            glob_cnt += 1

            optimizer_mlm.zero_grad()
            outputs = model_mlm(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer_mlm.step()

            if glob_cnt % 500 == 0:

                end = time.time()
                elapsed = end-start

                train_loss = loss.cpu().item()
                print('[TRAIN MLM {:0>4d}, {:0>4d}, {:0>5d} / {:0>4d}] loss: {:.6f}, elapsed {:.3f} secs'.format(i+1, epoch+1, glob_cnt, config.n_epochs_mlm, train_loss, elapsed))

                start = time.time()
                valid_loss = evaluate(model_mlm, evalloader_mlm)
                end = time.time()
                elapsed = end-start
                print('[VALID MLM {:0>4d}, {:0>4d}, {:0>5d} / {:0>4d}] loss: {:.6f}, elapsed {:.3f} secs'.format(i+1, epoch+1, glob_cnt, config.n_epochs_mlm, valid_loss, elapsed))

                # record result for every epoch
                dict_record['batch'].append(i+1)
                dict_record['epoch'].append(epoch+1)
                dict_record['glob_cnt'].append(glob_cnt)
                dict_record['train_loss'].append(train_loss)
                dict_record['valid_loss'].append(valid_loss)

                df_record = pd.DataFrame(dict_record)
                df_record.to_csv(MODEL_MLM_DIR + 'record.csv')

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    model_mlm.save_adapter(save_directory=MODEL_MLM_DIR + 'mlm_adapter/', adapter_name = 'mlm_adapter', with_head = False)
                    torch.save({
                        'i': i+1,
                        'epoch': epoch+1,
                        'glob_cnt': glob_cnt,
                        'optimizer_state_dict':optimizer_mlm.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    }, MODEL_MLM_DIR+'model_mlm.pth')

                    print('checkpoint saved')
                
                start = time.time()
        epoch_end = time.time()
        print('training time for one epoch:', epoch_end-epoch_start)


def train_cls_baseline(config):
    sub_model_dir = 'model_' + config.baseline_with_adapter + '_' +str(int(time.time()))
    MODEL_CLS_DIR = 'models_cls_baseline_' + config.model_size + '/'+ sub_model_dir + '/'
    tokenizer, model_cls = init_tokenizer_model(config)

    if config.baseline_with_adapter == 'no':
        model_cls.add_classification_head('cls')
    elif config.baseline_with_adapter == 'yes':
        model_cls.add_adapter('cls_adapter', set_active = True)
        model_cls.add_classification_head('cls')

    model_cls.to(C.DEVICE)

    # load data
    train_cls  = pd.read_csv(C.DATA_DIR+C.TRAIN_CLS_CSV, index_col=0)
    eval_cls  = pd.read_csv(C.DATA_DIR+C.EVAL_CLS_CSV, index_col=0)

    trainset_cls = CLSDataset(train_cls, tokenizer)
    evalset_cls = CLSDataset(eval_cls, tokenizer)

    trainloader_cls = torch.utils.data.DataLoader(trainset_cls, batch_size = 16, shuffle = True)
    evalloader_cls = torch.utils.data.DataLoader(evalset_cls, batch_size = 16, shuffle = True)

    optimizer_cls = AdamW(model_cls.parameters(), lr = config.lr_cls, weight_decay = config.wd_cls)

    best_val_loss = 10000

    # save config file
    if not os.path.exists(MODEL_CLS_DIR + 'cls_adapter/'):
            os.makedirs(MODEL_CLS_DIR+ 'cls_adapter/') 
    if not os.path.exists(MODEL_CLS_DIR + 'cls_adapter_head/'):
            os.makedirs(MODEL_CLS_DIR+ 'cls_adapter_head/') 
    config.to_json(MODEL_CLS_DIR + 'config.json')

    dict_record = {'batch': [], 'epoch':[], 'glob_cnt':[], 'train_loss': [], 'valid_loss': []}

    glob_cnt = 0
    
    start = time.time()
    for epoch in range(config.n_epochs_cls):
        epoch_start = time.time()
        for i, batch in enumerate(trainloader_cls):
            glob_cnt += 1
            optimizer_cls.zero_grad()
            outputs = model_cls(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer_cls.step()

            if glob_cnt % 200 == 0:

                end = time.time()
                elapsed = end-start
                train_loss = loss.cpu().item()
                print('[TRAIN CLS {:0>4d}, {:0>4d}, {:0>5d} / {:0>4d}] loss: {:.6f}, elapsed {:.3f} secs'.format(i+1, epoch+1, glob_cnt, config.n_epochs_cls, train_loss, elapsed))

                start = time.time()
                valid_loss = evaluate(model_cls, evalloader_cls)
                end = time.time()
                elapsed = end-start
                print('[VALID CLS {:0>4d}, {:0>4d}, {:0>5d} / {:0>4d}] loss: {:.6f}, elapsed {:.3f} secs'.format(i+1, epoch+1, glob_cnt, config.n_epochs_cls, valid_loss, elapsed))

                # record result for every epoch
                dict_record['batch'].append(i+1)
                dict_record['epoch'].append(epoch+1)
                dict_record['glob_cnt'].append(glob_cnt)
                dict_record['train_loss'].append(train_loss)
                dict_record['valid_loss'].append(valid_loss)

                df_record = pd.DataFrame(dict_record)
                df_record.to_csv(MODEL_CLS_DIR + 'record.csv')

                # save the best model
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss

                    torch.save(model_cls.state_dict(), MODEL_CLS_DIR + 'state_dict.pth')
                    if config.baseline_with_adapter == 'yes':
                        model_cls.save_adapter(save_directory=MODEL_CLS_DIR + 'cls_adapter/', adapter_name = 'cls_adapter', with_head = True)
                    model_cls.save_head(save_directory=MODEL_CLS_DIR + 'cls_adapter_head/', head_name = 'cls')

                    torch.save({
                        'i': i+1,
                       'epoch': epoch+1,
                        'glob_cnt': glob_cnt,
                        'optimizer_state_dict':optimizer_cls.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    }, MODEL_CLS_DIR+'model_cls.pth')

                    print('checkpoint saved')

                start = time.time()

        epoch_end = time.time()
        print('training time for one epoch:', epoch_end-epoch_start)
    return sub_model_dir


def train_cls_enhanced(config):
    sub_model_dir = 'model_' +str(int(time.time()))
    MODEL_MLM_DIR = 'models_mlm_' + config.model_size + '/' + config.mlm_adapter_name + '/'
    MODEL_CLS_DIR = 'models_cls_enhanced_' + config.model_size + '/' + sub_model_dir + '/'

    tokenizer, model_cls = init_tokenizer_model(config)

    model_cls.load_adapter(adapter_name_or_path=MODEL_MLM_DIR + 'mlm_adapter/', load_as = 'cls_adapter', set_active = True)
    model_cls.add_classification_head('cls')
    model_cls.to(C.DEVICE)

    # if config.update_adapter_cls == 'no':
    #     adapter_not_update_cls(model_cls)

    train_cls  = pd.read_csv(C.DATA_DIR+C.TRAIN_CLS_CSV, index_col=0)
    eval_cls  = pd.read_csv(C.DATA_DIR+C.EVAL_CLS_CSV, index_col=0)

    trainset_cls = CLSDataset(train_cls, tokenizer)
    evalset_cls = CLSDataset(eval_cls, tokenizer)

    trainloader_cls = torch.utils.data.DataLoader(trainset_cls, batch_size = 16, shuffle = True)
    evalloader_cls = torch.utils.data.DataLoader(evalset_cls, batch_size = 16, shuffle = True)

    optimizer_cls = AdamW(model_cls.parameters(), lr = config.lr_cls, weight_decay = config.wd_cls)

    best_val_loss = 10000

    # save config file
    if not os.path.exists(MODEL_CLS_DIR + 'cls_adapter/'):
            os.makedirs(MODEL_CLS_DIR+ 'cls_adapter/') 
    if not os.path.exists(MODEL_CLS_DIR + 'cls_adapter_head/'):
            os.makedirs(MODEL_CLS_DIR+ 'cls_adapter_head/') 
    config.to_json(MODEL_CLS_DIR + 'config.json')

    dict_record = {'batch': [], 'epoch':[], 'glob_cnt':[], 'train_loss': [], 'valid_loss': [], 'lr_cls': []}

    glob_cnt = 0

    start = time.time()
    for epoch in range(config.n_epochs_cls):
        epoch_start = time.time()
        for i, batch in tqdm(enumerate(trainloader_cls)):
            glob_cnt += 1
            optimizer_cls.zero_grad()
            outputs = model_cls(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer_cls.step()

            if glob_cnt % 200 == 0:

                end = time.time()
                elapsed = end-start
                train_loss = loss.cpu().item()
                print('[TRAIN CLS {:0>4d}, {:0>4d}, {:0>5d} / {:0>4d}] loss: {:.6f}, elapsed {:.3f} secs'.format(i+1, epoch+1, glob_cnt, config.n_epochs_cls, train_loss, elapsed))

                start = time.time()
                valid_loss = evaluate(model_cls, evalloader_cls)
                end = time.time()
                elapsed = end-start
                print('[VALID CLS {:0>4d}, {:0>4d}, {:0>5d} / {:0>4d}] loss: {:.6f}, elapsed {:.3f} secs'.format(i+1, epoch+1, glob_cnt, config.n_epochs_cls, valid_loss, elapsed))

                # record result for every epoch
                dict_record['batch'].append(i+1)
                dict_record['epoch'].append(epoch+1)
                dict_record['glob_cnt'].append(glob_cnt)
                dict_record['train_loss'].append(train_loss)
                dict_record['valid_loss'].append(valid_loss)
                # dict_record['lr_cls'].append(lr_cls)

                df_record = pd.DataFrame(dict_record)
                df_record.to_csv(MODEL_CLS_DIR + 'record.csv')

                # save the best model
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    torch.save(model_cls.state_dict(), MODEL_CLS_DIR + 'state_dict.pth')
                    model_cls.save_adapter(save_directory=MODEL_CLS_DIR + 'cls_adapter/', adapter_name = 'cls_adapter', with_head = True)
                    model_cls.save_head(save_directory=MODEL_CLS_DIR + 'cls_adapter_head/', head_name = 'cls')
                    torch.save({
                        'i': i+1,
                       'epoch': epoch+1,
                        'glob_cnt': glob_cnt,
                        'optimizer_state_dict':optimizer_cls.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    }, MODEL_CLS_DIR+'model_cls.pth')
                start = time.time()
        epoch_end = time.time()
        print('training time for one epoch:', epoch_end-epoch_start)
    
    return sub_model_dir

if __name__ == '__main__':
    print(C.DEVICE)
    config = Configuration.parse_cmd()
    train_mlm(config)
    # train_cls(config)






# def train_cls(config):
#     MODEL_MLM_DIR = 'models_mlm/' + config.mlm_adapter_name + '/'
#     MODEL_CLS_DIR = 'models_cls/model_' + config.is_baseline + '_' + config.baseline_with_adapter + '_' +str(C.TIME) + '/'
#     print(config.is_baseline)
#     model_cls = BertAdapterModel.from_pretrained(C.MODEL_NAME)
#     if config.is_baseline == 'no':
#         model_cls.load_adapter(adapter_name_or_path=MODEL_MLM_DIR + 'mlm_adapter/', load_as = 'cls_adapter', set_active = True)
#     elif config.is_baseline == 'yes':
#         if config.baseline_with_adapter == 'no':
#             model_cls.load_adapter(adapter_name_or_path=MODEL_MLM_DIR + 'mlm_adapter/', load_as = 'cls_adapter', set_active = False)
#         elif config.baseline_with_adapter == 'yes':
#             model_cls.add_adapter('cls_adapter', set_active = True)
        
#     model_cls.add_classification_head('cls')
#     model_cls.to(C.DEVICE)

#     if config.is_baseline == 'no' and config.update_adapter_cls == 'no':
#         adapter_not_update_cls(model_cls)

#     train_cls  = pd.read_csv(C.DATA_DIR+C.TRAIN_CLS_CSV, index_col=0)
#     eval_cls  = pd.read_csv(C.DATA_DIR+C.EVAL_CLS_CSV, index_col=0)

#     trainset_cls = CLSDataset(train_cls, tokenizer)
#     evalset_cls = CLSDataset(eval_cls, tokenizer)

#     trainloader_cls = torch.utils.data.DataLoader(trainset_cls, batch_size = 16, shuffle = True)
#     evalloader_cls = torch.utils.data.DataLoader(evalset_cls, batch_size = 16, shuffle = True)

#     optimizer_cls = AdamW(model_cls.parameters(), lr = config.lr_cls)

#     best_val_loss = 10000

#     # save config file
#     if not os.path.exists(MODEL_CLS_DIR + 'cls_adapter/'):
#             os.makedirs(MODEL_CLS_DIR+ 'cls_adapter/') 
#     config.to_json(MODEL_CLS_DIR + 'config.json')

#     dict_record = {'epoch':[], 'train_loss': [], 'valid_loss': [], 'lr_cls': []}
#     for epoch in range(config.n_epochs_cls):
#         loss_agg = 0
#         cnt = 0
#         start = time.time()
#         for i, batch in tqdm(enumerate(trainloader_cls)):

#             optimizer_cls.zero_grad()
#             outputs = model_cls(**batch)
#             loss = outputs.loss

#             loss.backward()
#             optimizer_cls.step()

#             loss_agg += loss.cpu().item() * len(batch)
#             cnt += len(batch)

#         train_loss = loss_agg/cnt
#         end = time.time()
#         elapsed = end-start

#         print('[TRAIN CLS {:0>5d} / {:0>3d}] loss: {:.6f}, elapsed {:.3f} secs'.format(epoch+1, config.n_epochs_cls, train_loss, elapsed))

#         start = time.time()
#         valid_loss = evaluate(model_cls, evalloader_cls)
#         end = time.time()
#         elapsed = end-start
#         print('[VALID CLS {:0>5d} / {:0>3d}] loss: {:.6f}, elapsed {:.3f} secs'.format(epoch+1, config.n_epochs_cls, valid_loss, elapsed))

#         for param_group in optimizer_cls.param_groups:
#             lr_cls = param_group['lr']

#         # record result for every epoch
#         dict_record['epoch'].append(epoch)
#         dict_record['train_loss'].append(train_loss)
#         dict_record['valid_loss'].append(valid_loss)
#         dict_record['lr_cls'].append(lr_cls)

#         df_record = pd.DataFrame(dict_record)
#         df_record.to_csv(MODEL_CLS_DIR + 'record.csv')

#         # save the best model
#         if valid_loss < best_val_loss:
#             best_val_loss = valid_loss
#             model_cls.save_adapter(save_directory=MODEL_CLS_DIR + 'cls_adapter/', adapter_name = 'cls_adapter', with_head = True)
#             torch.save({
#                 'epoch': epoch,
#                 'optimizer_state_dict':optimizer_cls.state_dict(),
#                 'train_loss': train_loss,
#                 'valid_loss': valid_loss
#             }, MODEL_CLS_DIR+'model_cls.pth')

