import random
import pandas as pd
from configuration import Configuration
from configuration import CONSTANTS as C
from data_augmentation import model_init, generate_store
from metrics import get_metric
import torch
import time
import warnings
warnings.filterwarnings("ignore")

#ITERATE WITH DICT
#最后也用dict 保存最后的csv结果
method_list = ['top_k','top_p']#'beam'
metric_list = ['bleu','perplexity','frequency']
dict_list = {'trigger_name':[], 'generate_name':[],'max_gen_len':[],'method':[],'top_k':[],'top_p':[],\
    'temperature':[],'num_return':[]}#'num_beams':[],
# dict_test = {'trigger_name':'trigger_data_new.csv', 'generate_name': 'test1.csv', 'model':'GPT2','bs': 16, 'max_gen_len': 50, 'method': 'top_k', 'top_k': 30, 'top_p': 0.5, 'num_beams': 5, 'temperature': 0.5, 'num_return': 6
# }
for i in range(400):
    # dict_test = {'trigger_name':'trigger_data_new.csv', 'generate_name': 'test1.csv', 'model':'GPT2','bs': 16, \
    # 'max_gen_len': 50, 'method': 'top_k', 'top_k': 30, 'top_p': 0.5, 'num_beams': 5, 'temperature': 0.5, 'num_return': 6}
    start = time.time()
    torch.cuda.empty_cache()
    try:
        trigger_name = 'rsearch_trigger.csv'
        generate_name = 'test_aug_2/test'+str(i)+'.csv'
        max_gen_len = random.randint(40,60)
        method = method_list[random.randint(0,1)]
        top_k = random.randint(20,40)
        top_p = 0.1*random.randint(0,10)
        # num_beams = random.randint(1,10)
        temperature = 0.1*random.randint(1,10)
        num_return = random.randint(1, 10)
        dict_temp = {'trigger_name': trigger_name, 'generate_name': generate_name, 'model':'GPT2','bs': 16, \
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
    # #metric:
    # for metric in metric_list:
    #     dict_temp_metric = {'metric':metric,"eval_name":generate_name,'refer_name':'slang_augment_30000_split.csv'}
    #     config = Configuration(dict_temp_metric)
    #     metric_cal = get_metric(config)
    #     dict_list[metric].append(metric_cal)

    except:
        print('outof memory??')

    end = time.time()
    print('time', int(end-start))

df = pd.DataFrame(dict_list)
df.to_csv(C.DATA_DIR+ "random_search_result.csv")