import torch
from configuration import CONSTANTS as C
from tkinter import _flatten
import pandas as pd

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append(list(range(ind,ind+sll)))

    return results

def data_cls_csv():
    train_sl = pd.read_csv(C.DATA_DIR + "slang_train_10000_split.csv")
    train_st = pd.read_csv(C.DATA_DIR + "standard_train_10000.csv")
    test_sl = pd.read_csv(C.DATA_DIR + "slang_test_10000_split.csv")
    test_st = pd.read_csv(C.DATA_DIR + "standard_test_10000.csv")

    train_sl["label"] = 1
    train_st["label"] = 0
    test_sl["label"] = 1
    test_st["label"] = 0

    train_sl = train_sl[['example', 'label']]
    train_st = train_st[['train', 'label']]
    test_sl = test_sl[['example', 'label']]
    test_st = test_st[['test', 'label']]

    train_st.columns = ['example', 'label']
    test_st.columns = ['example', 'label']

    eval_sl = test_sl[:5000]
    eval_st = test_st[:5000]
    test_sl = test_sl[5000:]
    test_st = test_st[5000:]

    trainset = pd.concat([train_sl,train_st], axis = 0).reset_index(drop = True)
    evalset = pd.concat([eval_sl,eval_st], axis = 0).reset_index(drop = True)
    testset = pd.concat([test_sl,test_st], axis = 0).reset_index(drop = True)

    trainset.to_csv(C.DATA_DIR + C.TRAIN_CLS_CSV)
    evalset.to_csv(C.DATA_DIR + C.EVAL_CLS_CSV)
    testset.to_csv(C.DATA_DIR + C.TEST_CLS_CSV)


class MLMDateset(torch.utils.data.Dataset):
    """
    Dataset for mask language modelling task.
    """
    def __init__(self, data, tokenizer, config):
        self.data = list(data['generate'])
        self.word = list(data['word'])
        self.tokenizer = tokenizer
        self.inputs = self.tokenizer(self.data, return_tensors="pt", padding=True) #pt: return pytorch tensor
        self.random_tensor = torch.rand(self.inputs['input_ids'].shape)
        self.random_tensor2 = torch.rand(self.inputs['input_ids'].shape)
        self.masked_tensor = None
        self.config = config
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        self.create_MLM()
        return {'input_ids':self.inputs['input_ids'].detach().clone()[idx].to(C.DEVICE),
                'token_type_ids':self.inputs['token_type_ids'].detach().clone()[idx].to(C.DEVICE),
                'attention_mask':self.inputs['attention_mask'].detach().clone()[idx].to(C.DEVICE),
                'labels':self.inputs['labels'].detach().clone()[idx].to(C.DEVICE)}

    def create_MLM(self):
        #get the index of slang word in each sentence
        index_result = []
        for i in range(len(self.data)):
            inputs = self.tokenizer(self.data[i], return_tensors="pt", padding=True)
            word = self.tokenizer(self.word[i], return_tensors="pt", padding=True)
            list1 = inputs['input_ids'].numpy().tolist()
            list2 = word['input_ids'].numpy().tolist()
            del(list2[0][len(list2[0])-1])
            del(list2[0][0])
            index_result.append(list(_flatten(find_sub_list(list2[0],list1[0]))))

        #create false tensor
        mask_tensor = torch.zeros(self.inputs['input_ids'].shape,dtype=torch.bool)
        for i in range(len(mask_tensor)):
            mask_tensor[i, index_result[i]] = True 

        self.inputs['labels'] = self.inputs['input_ids'].detach().clone() 
        self.masked_tensor = (self.random_tensor2 < self.config.mlm_threshold) *(self.random_tensor < 0.15)*(self.inputs['input_ids'] != '101') * \
            (self.inputs['input_ids'] != '102') * (self.inputs['input_ids'] != 0)\
                | (self.random_tensor2 >= self.config.mlm_threshold) * mask_tensor

    
        non_zero_indices = []
        for i in range(len(self.inputs['input_ids'])):
            non_zero_indices.append(torch.flatten(self.masked_tensor[i].nonzero()).tolist())#Flattens input by reshaping it into a one-dimensional tensor. 
        for i in range(len(self.inputs['input_ids'])):
            self.inputs['input_ids'][i, non_zero_indices[i]] = 103 #103: masked token


class CLSDataset(torch.utils.data.Dataset):
    """
    Dataset for classification task.
    """
    def __init__(self, data_cls, tokenizer):
        self.data_cls  = data_cls
        self.tokenizer = tokenizer
        self.inputs = tokenizer(list(self.data_cls['example']), return_tensors="pt", padding=True)
        self.inputs['labels'] = torch.tensor(self.data_cls['label'])
    def __len__(self):
        return len(self.data_cls)
    def __getitem__(self, idx):
        return {'input_ids':self.inputs['input_ids'].detach().clone()[idx].to(C.DEVICE),
                'token_type_ids':self.inputs['token_type_ids'].detach().clone()[idx].to(C.DEVICE),
                'attention_mask':self.inputs['attention_mask'].detach().clone()[idx].to(C.DEVICE),
                'labels':self.inputs['labels'].detach().clone()[idx].to(C.DEVICE)}
