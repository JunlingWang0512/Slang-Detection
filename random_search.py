import random
import pandas as pd
from configuration import Configuration
from configuration import CONSTANTS as C
from data_augmentation import model_init, generate_store
from metrics import get_metric
import torch
import time
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def rsearch_trigger_csv():
    filedir = C.DATA_DIR + C.AUG_TRIGGER_CSV
    trigger_new = pd.read_csv(filedir, index_col=0)
    random.seed(122)
    sample_idx = random.sample(range(0, trigger_new.shape[0]), k=1000)
    rsearch_trigger = trigger_new.iloc[sample_idx].reset_index(drop = True)
    rsearch_trigger.to_csv(C.DATA_DIR + 'rsearch_trigger.csv')


def random_search_paras():
    #ITERATE WITH DICT
    #最后也用dict 保存最后的csv结果
    method_list = ['top_k','top_p']#'beam'
    metric_list = ['bleu','perplexity','frequency']
    dict_list = {'trigger_name':[], 'generate_name':[],'max_gen_len':[],'method':[],'top_k':[],'top_p':[],\
        'temperature':[],'num_return':[]}#'num_beams':[],
  
    for i in range(400):
        # dict_test = {'trigger_name':'trigger_data_new.csv', 'generate_name': 'test1.csv', 'model':'GPT2','bs': 16, \
        # 'max_gen_len': 50, 'method': 'top_k', 'top_k': 30, 'top_p': 0.5, 'num_beams': 5, 'temperature': 0.5, 'num_return': 6}
        start = time.time()
        torch.cuda.empty_cache()
        try:
            if not os.path.exists(C.DATA_DIR+'test_aug/'):
                os.makedirs(C.DATA_DIR+'test_aug/') 
            generate_name = 'test_aug/test'+str(i)+'.csv'
            max_gen_len = random.randint(40,60)
            method = method_list[random.randint(0,1)]
            top_k = random.randint(20,40)
            top_p = 0.1*random.randint(0,10)
            # num_beams = random.randint(1,10)
            temperature = 0.1*random.randint(1,10)
            num_return = random.randint(1, 10)
            dict_temp = {'generate_name': generate_name, 'model':'GPT2','bs': 16, \
            'max_gen_len': max_gen_len, 'method': method, 'top_k': top_k, 'top_p':top_p , 'temperature':temperature , 'num_return': num_return} #'num_beams': num_beams,
            print(dict_temp)
            config = Configuration(dict_temp)
            tokenizer, model = model_init(config)
            df_trigger = pd.read_csv(C.DATA_DIR + config.trigger_name, index_col = 0)
            generate_store(df_trigger, tokenizer, model, config)

            #append
            for key in dict_list.keys():
                if key not in metric_list:
                    dict_list[key].append(dict_temp[key])

        except:
            print('outof memory??')

        end = time.time()
        print('time', int(end-start))

    df = pd.DataFrame(dict_list)
    df.to_csv(C.DATA_DIR+ "random_search_result.csv")


def metric_cal():
    #metric:
    random_search_df = pd.read_csv(C.DATA_DIR + 'random_search_result.csv', index_col = 0)
    metric_dict_list = {'bleu':[], 'perplexity':[], 'count':[], 'count_rate':[]}
    # metric_dict_list = {'count':[], 'count_rate':[]}
    metric_list = ['bleu','perplexity','frequency']
    # metric_list = ['frequency']
    for i in range(len(random_search_df)):
        print('i =', i)
        generate_name = random_search_df.loc[i, 'generate_name']
        for metric in metric_list:
            dict_temp_metric = {'metric':metric,"eval_name":generate_name,'refer_name':'slang_augment_50000_updated.csv'}
            config = Configuration(dict_temp_metric)
            metric_cal = get_metric(config)
            if metric == 'bleu':
                metric_dict_list[metric].append(metric_cal['bleu'])
                print(metric_cal['bleu'])
            if metric == 'perplexity':
                metric_dict_list[metric].append(metric_cal['mean_perplexity'])
                print(metric_cal['mean_perplexity'])
            if metric == 'frequency':
                metric_dict_list['count'].append(metric_cal[0])
                print(metric_cal[0])
                metric_dict_list['count_rate'].append(metric_cal[1])
                print(metric_cal[1])
        if i%20 == 0:
            df = pd.DataFrame(metric_dict_list)
            df.to_csv(C.DATA_DIR+'random_search_metrics.csv')

    df = pd.DataFrame(metric_dict_list)
    df.to_csv(C.DATA_DIR+'random_search_metrics.csv')


def human_eval_csv(metric_name):
    df_metrics = pd.read_csv(C.DATA_DIR+'random_search_metrics.csv', index_col = 0)
    df_paras = pd.read_csv(C.DATA_DIR+'random_search_result.csv', index_col = 0)

    max_idx = np.argmax(np.array(df_metrics[metric_name]))
    new_metrics = abs(np.array(df_metrics[metric_name]) - np.mean(np.array(df_metrics[metric_name])))
    avg_idx = np.argmin(new_metrics)
    min_idx = np.argmin(np.array(df_metrics[metric_name]))
    

    df_max = pd.read_csv(C.DATA_DIR+df_paras.loc[max_idx, 'generate_name'], index_col=0)
    df_avg = pd.read_csv(C.DATA_DIR+df_paras.loc[min_idx, 'generate_name'], index_col=0)
    df_min = pd.read_csv(C.DATA_DIR+df_paras.loc[avg_idx, 'generate_name'], index_col=0)

    word_set = set(df_max['word']).intersection(df_avg['word']).intersection(df_min['word'])

    df_word_max = df_max.loc[df_max['word'].isin(word_set)].sort_values(by = 'word').reset_index(drop = True)
    df_word_avg = df_avg.loc[df_avg['word'].isin(word_set)].sort_values(by = 'word').reset_index(drop = True)
    df_word_min = df_min.loc[df_min['word'].isin(word_set)].sort_values(by = 'word').reset_index(drop = True)

    df_word_max.to_csv(C.DATA_DIR + 'human_eval/' +metric_name + '_max.csv')
    df_word_avg.to_csv(C.DATA_DIR + 'human_eval/' +metric_name + '_avg.csv')
    df_word_min.to_csv(C.DATA_DIR + 'human_eval/' +metric_name + '_min.csv')


def avg_para_augment():
    df_metrics = pd.read_csv(C.DATA_DIR+'random_search_metrics.csv', index_col = 0)
    df_paras = pd.read_csv(C.DATA_DIR+'random_search_result.csv', index_col = 0)

    metric_list = ['bleu', 'perplexity', 'count_rate']
    # metric_list = ['perplexity', 'count_rate']
    for metric_name in metric_list:
        # max_idx = np.argmax(np.array(df_metrics[metric_name]))
        new_metrics = abs(np.array(df_metrics[metric_name]) - np.mean(np.array(df_metrics[metric_name])))
        avg_idx = np.argmin(new_metrics)
        # min_idx = np.argmin(np.array(df_metrics[metric_name]))

        if not os.path.exists(C.DATA_DIR+'aug_final/'):
            os.makedirs(C.DATA_DIR+'aug_final/') 
        dict_avg_paras = dict(df_paras.loc[avg_idx])
        dict_avg_paras['generate_name'] = 'aug_final/avg_'+metric_name + '.csv'
        dict_avg_paras['trigger_name'] = C.AUG_TRIGGER_CSV
        dict_avg_paras['model'] = 'GPT2'
        dict_avg_paras['bs'] = 16

        config = Configuration(dict_avg_paras)
        tokenizer, model = model_init(config)
        df_trigger = pd.read_csv(C.DATA_DIR + config.trigger_name, index_col = 0)
        generate_store(df_trigger, tokenizer, model, config)


def joined_augment():
    metric_list = ['bleu', 'perplexity', 'count_rate']
    df_list = []
    for metric_name in metric_list:
        df_list.append(pd.read_csv(C.DATA_DIR + 'aug_final/avg_'+metric_name + '.csv', index_col = 0))
    joined_augment = pd.concat(df_list, axis = 0).sort_values(by = 'word').drop_duplicates(keep='first').reset_index(drop = True).drop(columns = ['index'])
    joined_augment.to_csv(C.DATA_DIR + C.AUG_RESULT_CSV)

if __name__ == '__main__':
    # random_search_paras()
    # metric_cal()

    # human_eval_csv('bleu')
    # human_eval_csv('perplexity')
    # human_eval_csv('count')
    # human_eval_csv('count_rate')
    avg_para_augment()