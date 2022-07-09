from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd
from tqdm import tqdm
import re
import random
import warnings
from configuration import Configuration
from configuration import CONSTANTS as C
warnings.filterwarnings("ignore")

# get device
print(C.DEVICE)

def model_init(config):
    if config.model == 'GPT2':
        # GPT2 tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2", return_dict = True).to(C.DEVICE)

        # GPT2 parameter setting
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    return tokenizer, model


def example_gener(word,wordlist):
    s= ""
#     print("word",word)
#     print("wordlist",wordlist)
    for i in range(len(wordlist)):
#         print("i",i)
#         print("wordlist[i]",wordlist[i])
        s = s+str(i+1) + ". " + str(word) + " : "+str(wordlist[i])+"\n"
    s = s + str(len(wordlist)+1) + ". " +str(word) + " : "
    return s


def data_trigger_csv():
    filedir = C.DATA_DIR + C.AUG_ORIGIN_CSV
    data_cleaned = pd.read_csv(filedir).sort_values(['word'])

    temp_list = []
    tempword = data_cleaned.iloc[0, 0]

    trigger_list = []
    trigger_word = []
    trigger_len = []
    for i in range(len(data_cleaned)):
        if (data_cleaned.iloc[i, 0] == tempword):
            temp_list.append(data_cleaned.iloc[i, 1])
        else:
            s = example_gener(tempword, temp_list)
            trigger_list.append(s)
            trigger_word.append(tempword)
            trigger_len.append(len(temp_list))
            tempword = data_cleaned.iloc[i, 0]
            temp_list = [data_cleaned.iloc[i, 1]]


    df_trigger = pd.DataFrame(columns = ['word', 'trigger', 'length'])
    df_trigger['word'] = trigger_word
    df_trigger['length'] = trigger_len
    df_trigger['trigger'] = trigger_list
    df_trigger = df_trigger.reset_index()
    df_trigger.to_csv(C.DATA_DIR+C.AUG_TRIGGER_CSV, index = False)


def model_batch_generation(batch_list, tokenizer, model, config):
    torch.cuda.empty_cache()
    inputs = tokenizer(batch_list, return_tensors="pt", padding=True)

    if inputs['input_ids'].size(1)>512:
        inputs['input_ids'] = inputs['input_ids'][:, -512:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -512:]
    if config.method == 'top_k':
        output_sequences = model.generate(
            input_ids=inputs['input_ids'].to(C.DEVICE),
            attention_mask=inputs['attention_mask'].to(C.DEVICE),
            do_sample=True, # disable sampling to test if batching affects output
            max_length=len(inputs['input_ids'][0]) + config.max_gen_len, # let it generate longer
            pad_token_id=tokenizer.eos_token_id,
            top_k = int(config.top_k),
            temperature=config.temperature,
            num_return_sequences= config.num_return
        )
    if config.method == 'top_p':
        output_sequences = model.generate(
            input_ids=inputs['input_ids'].to(C.DEVICE),
            attention_mask=inputs['attention_mask'].to(C.DEVICE),
            do_sample=True, # disable sampling to test if batching affects output
            max_length=len(inputs['input_ids'][0]) + config.max_gen_len, # let it generate longer
            pad_token_id=tokenizer.eos_token_id,
            top_p = config.top_p,
            temperature=config.temperature,
            num_return_sequences= config.num_return
        )
    if config.method == 'beam':
        output_sequences = model.generate(
            input_ids=inputs['input_ids'].to(C.DEVICE),
            attention_mask=inputs['attention_mask'].to(C.DEVICE),
            do_sample=True, # disable sampling to test if batching affects output
            max_length=len(inputs['input_ids'][0]) + config.max_gen_len, # let it generate longer
            pad_token_id=tokenizer.eos_token_id,
            num_beams = config.num_beams,
            temperature=config.temperature,
            num_return_sequences= config.num_return
        )

    outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in output_sequences]
    return outputs


def extract_sent(s, word, num):
    s2 = ''
    # print(s)
    # print(word)
    # print(num)
    if(len(re.findall(str(num+1)+". "+str(word)+" :(.*?)\n", s)) >= 1):
        s1 = re.findall(str(num+1)+". "+str(word)+" :(.*?)\n", s)[0]
        if(len(re.findall("(.*?)"+str(word)+"(.*?)",s1)) >= 1):
            s2 = s1.replace('\xa0', '')
            s2 = s2.strip()
            if len(re.findall("[0-9]+\. "+str(word)+" :(.*?)$", s2)) >= 1:
                s2 = re.findall("(.*?)[0-9]+\. "+str(word)+" :", s2)[0].strip()
    return s2


def generate_store(tokenizer, model, config):
    # generate result storage
    df_trigger = pd.read_csv(C.DATA_DIR + C.AUG_TRIGGER_CSV, index_col = 0)

    succeed = {'trigger': [], 'word': [], 'generate':[]}
    trigger_list = []
    word_list = []
    generated_list = []


    for i in tqdm(range(0, len(df_trigger), config.bs)):
    # for i in range(144, len(df_trigger), batchsize):
        # print(i)
        # generation setting
        batch = df_trigger.loc[i: i+config.bs-1, :]
        batch_word = list(batch['word'])
        batch_list = list(batch['trigger'])
        batch_lens = list(batch['length'])

        # generate sentences
        outputs = model_batch_generation(batch_list, tokenizer, model, config)


        # extract sentences for each word
        for j in range(len(batch)):
            trigger = batch_list[j]
            word = batch_word[j]
            num = batch_lens[j]
            for k in range(config.num_return*j, config.num_return*j+config.num_return):
                try:
                    s = extract_sent(str(outputs[k]), word, num)
                except:
                    s = ''
                    print(i,j)
                if s != '':
                    trigger_list.append(trigger)
                    word_list.append(word)
                    generated_list.append(s)

        # storage
        if i%100 == 0 :
            succeed['trigger'] = trigger_list
            succeed['word'] = word_list
            succeed['generate'] = generated_list
            df_succeed = pd.DataFrame(succeed)
            df_succeed = df_succeed.drop_duplicates(keep='first').reset_index()
            df_succeed.to_csv(C.DATA_DIR+config.generate_name)


def augment_split_csv(config):
    filedir = C.DATA_DIR + config.aug_result_csv
    data_augment = pd.read_csv(filedir, index_col=0)  
    random.seed(122)
    sample_idx = random.sample(range(0, data_augment.shape[0]), k=data_augment.shape[0])
    train_cnt = int(data_augment.shape[0]* 0.8)
    train = data_augment.iloc[sample_idx[:train_cnt]]
    eval = data_augment.iloc[sample_idx[train_cnt:]]
    train.to_csv(C.DATA_DIR + C.TRAIN_MLM_CSV)
    eval.to_csv(C.DATA_DIR + C.EVAL_MLM_CSV)



if __name__ == '__main__':
    config = Configuration.parse_cmd()
    data_trigger_csv()
    print("trigger data generated")
    tokenizer, model = model_init(config)
    # load trigger data file
    generate_store(tokenizer, model, config)



