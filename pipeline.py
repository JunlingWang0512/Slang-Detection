import pandas as pd
from configuration import Configuration
from configuration import CONSTANTS as C
from data_augmentation import data_trigger_csv, model_init, generate_store
from random_search import rsearch_trigger_csv, random_search_paras, metric_cal, human_eval_csv, avg_para_augment, joined_augment
from dataset_mlm_cls import augment_split_csv, data_cls_csv
from train import train_mlm, train_cls_baseline, train_cls_enhanced

if __name__=='__main__':
    config = Configuration.parse_cmd()

    # ******************* extraction and preprocessing *****************
    # use the file data_processing.py


    # *******************  augmentation **************
    if config.pipeline == 'data_augmentation':
        # generate the triggering csv file
        data_trigger_csv()
        print("trigger data generated")

        # generate augmentation data
        tokenizer, model = model_init(config)
        generate_store(tokenizer, model, config)

    if config.pipeline == 'augment_random_search':
        # random search extract triggering data
        rsearch_trigger_csv()
        # random search different parameters augmentation
        random_search_paras()
        # calculate random search metrics
        metric_cal()

    if config.pipeline == 'random_search_human_eval':
        # human evalulation extraction for random search
        human_eval_csv('bleu')
        human_eval_csv('perplexity')
        human_eval_csv('count_rate')

    if config.pipeline == 'final_augmentation':
        # must have random_search_metrics.csv and random_serach_result.csv
        # averate parameter augmentation
        avg_para_augment()
        # join all the augmentation results
        joined_augment()
        # split augment result into mlm training and validation sets
        augment_split_csv()


    # ******************* mlm *****************
    if config.pipeline == 'mlm_train':
        # train mlm
        train_mlm(config)

    # ******************* cls *****************
    if config.pipeline == 'cls_data_generation':
        data_cls_csv()

    if config.pipeline == 'cls_baseline':
        train_cls_baseline(config)

    if config.pipeline == 'cls_enhanced':
        train_cls_enhanced(config)
    

        
