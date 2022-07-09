import pandas as pd
from configuration import Configuration
from configuration import CONSTANTS as C
from data_extraction import extraction_csv
from data_augmentation import data_trigger_csv, model_init, generate_store 
from random_search import rsearch_trigger_csv, random_search_paras, metric_cal, human_eval_csv, avg_para_augment

if __name__=='__main__':
    config = Configuration.parse_cmd()

    if config.pipeline == 'data_extraction':
        extraction_csv()

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
        avg_para_augment()

    if config.pipeline == 'random_search_train':
        print("random_search_train")
    

        
