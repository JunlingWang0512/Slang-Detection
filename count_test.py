import torch
import pandas as pd
from configuration import CONSTANTS as C
from configuration import Configuration
from dataset_mlm_cls import MLMDateset, CLSDataset
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

    testset_cls = CLSDataset(test_cls, tokenizer)
    testloader_cls = torch.utils.data.DataLoader(testset_cls, batch_size = 16, shuffle = True)

    test_loss = evaluate(model_cls_test, testloader_cls)
    print(test_loss)
    return test_loss

def count_test_enhanced_cls(config):
    print(C.DEVICE)
    TEST_MODEL_DIR = 'models_cls_enhanced_'+ config.model_size + '/' +config.test_model_dir +'/'
    print(TEST_MODEL_DIR)
    tokenizer, model_cls_test = init_tokenizer_model(config)
    
    model_cls_test.add_adapter('cls_adapter', set_active = True)
    model_cls_test.add_classification_head('cls')
    model_cls_test.load_state_dict(torch.load(TEST_MODEL_DIR+'state_dict.pth'))
    model_cls_test.to(C.DEVICE)

    test_data = pd.read_csv('data/slang_test.csv')
    test_data['label'] = 1
    test_group = test_data.groupby('count')
    key_list = []
    test_loss_list = []
    for key, df_group in test_group:
        testset_cls = CLSDataset(df_group, tokenizer)
        testloader_cls = torch.utils.data.DataLoader(testset_cls, batch_size = 16, shuffle = True)
        test_loss = evaluate(model_cls_test, testloader_cls)
        print(key, test_loss)
        key_list.append(key)
        test_loss_list.append(test_loss)
    return key_list, test_loss_list


if __name__ == '__main__':
    model_size = 'base'
    model_dir = 'model_1657724797'

    dict_test = {'model_size': model_size, 'test_model_dir': model_dir}
    config = Configuration(dict_test)
    count_test_enhanced_cls(config)