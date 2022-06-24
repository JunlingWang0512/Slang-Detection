from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd
from tqdm import tqdm
import re
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

def model_batch_generation(batch_list, tokenizer, model, config):

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
            top_k = config.top_k,
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
    return s2


def generate_store(df_trigger, tokenizer, model, config):
    # generate result storage
    succeed = {'trigger': [], 'word': [], 'generate':[]}
    trigger_list = []
    word_list = []
    generated_list = []


    for i in tqdm(range(4192, len(df_trigger), config.bs)):
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
            df_succeed.to_csv(C.DATA_DIR+config.generate_name)



if __name__ == '__main__':
    config = Configuration.parse_cmd()
    tokenizer, model = model_init(config)
    # load trigger data file
    df_trigger = pd.read_csv(C.DATA_DIR + config.trigger_name, index_col = 0)
    generate_store(df_trigger, tokenizer, model, config)



