import pandas as pd
from configuration import Configuration
from configuration import CONSTANTS as C
from data_extraction import extraction_csv
from data_augmentation import data_trigger_csv, model_init, generate_store
from random_search import rsearch_trigger_csv, random_search_paras, metric_cal, human_eval_csv, avg_para_augment, joined_augment
from dataset_mlm_cls import augment_split_csv, data_cls_csv
from train import train_mlm

if __name__=='__main__':
    config = Configuration.parse_cmd()

    # ******************* extraction *****************
    if config.pipeline == 'data_extraction':
        extraction_csv()


    # *******************  augmentation **************
    if config.pipeline == 'data_augmentation':
        data_trigger_csv()
        print("trigger data generated")
        tokenizer, model = model_init(config)
        generate_store(tokenizer, model, config)

    if config.pipeline == 'augment_random_search':
        rsearch_trigger_csv()
        random_search_paras()
        metric_cal()

    if config.pipeline == 'random_search_human_eval':
        human_eval_csv('bleu')
        human_eval_csv('perplexity')
        human_eval_csv('count_rate')

    if config.pipeline == 'final_augmentation':
        # must have random_search_metrics.csv and random_serach_result.csv
        # avg_para_augment()
        joined_augment()
        augment_split_csv()


    # ******************* mlm *****************
    if config.pipeline == 'mlm_train':
        train_mlm(config)
    
    if config.pipeline == 'random_search_mlm':
        print('random_search_mlm')

    # ******************* cls *****************
    if config.pipeline == 'cls_data_generation':
        data_cls_csv()

    if config.pipeline == 'random_search_cls':
        print("random_search_cls")
    

        
