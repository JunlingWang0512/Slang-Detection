import torch
from configuration import CONSTANTS as C

class MLMDateset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data  = data
        self.inputs = tokenizer(self.data, return_tensors="pt", padding=True)
        self.random_tensor = torch.rand(self.inputs['input_ids'].shape)
        self.masked_tensor = None
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        self.create_MLM()
        return {key: torch.tensor(val[idx]).to(C.DEVICE) for key, val in self.inputs.items()}

    def create_MLM(self):
        self.inputs['labels'] = self.inputs['input_ids'].detach().clone()
        self.masked_tensor = (self.random_tensor < 0.15)*(self.inputs['input_ids'] != '101') * (self.inputs['input_ids'] != '102') * (self.inputs['input_ids'] != 0)
        non_zero_indices = []
        for i in range(len(self.inputs['input_ids'])):
            non_zero_indices.append(torch.flatten(self.masked_tensor[i].nonzero()).tolist())
        for i in range(len(self.inputs['input_ids'])):
            self.inputs['input_ids'][i, non_zero_indices[i]] = 103


class CLSDataset(torch.utils.data.Dataset):
    def __init__(self, data_cls, tokenizer):
        self.data_cls  = data_cls
        self.tokenizer = tokenizer
        self.inputs = tokenizer(list(self.data_cls['example']), return_tensors="pt", padding=True)
        self.inputs['labels'] = torch.tensor(self.data_cls['label'])
    def __len__(self):
        return len(self.data_cls)
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]).to(C.DEVICE) for key, val in self.inputs.items()}