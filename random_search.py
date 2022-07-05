import random
import pandas as pd
from configuration import Configuration
from data_augmentation import model_init, generate_store
from metrics import get_metric

#ITERATE WITH DICT
#最后也用dict 保存最后的csv结果
method_list = ['top_k','top_p','beam']
metric_list = ['bleu','perplexity','frequency']
dict_list = {'trigger_name':[], 'generate_name':[],'max_gen_len':[],'method':[],'top_k':[],'top_p':[],\
    'num_beams':[],'temperature':[],'num_return':[],'bleu':[],'perplexity':[],'freqency':[]}
# dict_test = {'trigger_name':'trigger_data_new.csv', 'generate_name': 'test1.csv', 'model':'GPT2','bs': 16, 'max_gen_len': 50, 'method': 'top_k', 'top_k': 30, 'top_p': 0.5, 'num_beams': 5, 'temperature': 0.5, 'num_return': 6
# }
for i in range(1000):
    # dict_test = {'trigger_name':'trigger_data_new.csv', 'generate_name': 'test1.csv', 'model':'GPT2','bs': 16, \
    # 'max_gen_len': 50, 'method': 'top_k', 'top_k': 30, 'top_p': 0.5, 'num_beams': 5, 'temperature': 0.5, 'num_return': 6}
    trigger_name = 'trigger_data_new.csv'
    generate_name = 'test'+str(i)+'.csv'
    max_gen_len = random.randint(40,60)
    method = method_list[random.randint(0,2)]
    top_k = random.randint(20,40)
    top_p = 0.1*random.randint(0,10)
    num_beams = random.randint(0,10)
    temperature = 0.1*random.randint(0,10)
    num_return = random.randint(0,10)
    dict_temp = {'trigger_name': trigger_name, 'generate_name': generate_name, 'model':'GPT2','bs': 16, \
    'max_gen_len': max_gen_len, 'method': method, 'top_k': top_k, 'top_p':top_p , 'num_beams': num_beams, 'temperature':temperature , 'num_return': num_return}
    # print(dict_test)
    config = Configuration(dict_temp)
    tokenizer, model = model_init(config)
    df_trigger = pd.read_csv('data/'+ config.trigger_name, index_col = 0)
    generate_store(df_trigger, tokenizer, model, config)

    #append
    for key in dict_list.keys() and key not in metric_list:
        dict_list[key].append(dict_temp[key])
    #metric:
    for metric in metric_list:
        dict_temp_metric = {'metric':metric,"eval_name":generate_name,'refer_name':'slang_augment_30000_split.csv'}
        config = Configuration(dict_temp_metric)
        dict_list[metric].append(get_metric(config))

df = pd.DataFrame.from_dict(dict_list, orient="index")
df.to_csv("data/random_search_result.csv")