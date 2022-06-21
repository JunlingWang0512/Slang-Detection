from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings("ignore")

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def GPT2_batch_generation(batch_list, max_gen_len, top_p, num_beams, method = 'top_k', top_k = 20, temperature = 0.5, num_return = 4):

    inputs = tokenizer(batch_list, return_tensors="pt", padding=True)

    if inputs['input_ids'].size(1)>512:
        inputs['input_ids'] = inputs['input_ids'][:, -512:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -512:]

    if method == 'top_k':
        output_sequences = model.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            do_sample=True, # disable sampling to test if batching affects output
            max_length=len(inputs['input_ids'][0]) + max_gen_len, # let it generate longer
            pad_token_id=tokenizer.eos_token_id,
            top_k = top_k,
            temperature=temperature,
            num_return_sequences= num_return
        )
    if method == 'top_p':
        output_sequences = model.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            do_sample=True, # disable sampling to test if batching affects output
            max_length=len(inputs['input_ids'][0]) + max_gen_len, # let it generate longer
            pad_token_id=tokenizer.eos_token_id,
            top_p = top_p,
            temperature=temperature,
            num_return_sequences= num_return
        )
    if method == 'beam':
        output_sequences = model.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            do_sample=True, # disable sampling to test if batching affects output
            max_length=len(inputs['input_ids'][0]) + max_gen_len, # let it generate longer
            pad_token_id=tokenizer.eos_token_id,
            num_beams = num_beams,
            temperature=temperature,
            num_return_sequences= num_return
        )

    outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in output_sequences]
    return outputs


def extract_sent(s, word, num):
    s2 = ''
    # print(s)
    # print(word)
    # print(num)
    if(len(re.findall(str(num+1)+". "+str(word)+" :(.*?)\n", s)) >= 1):
        s2 = re.findall(str(num+1)+". "+str(word)+" :(.*?)\n", s)[0]
        if(len(re.findall("(.*?)"+str(word)+"(.*?)",s2)) >= 1):
            s2 = s2.replace('\xa0', '')
            s2 = s2.strip()
    return s2


# parameters
batchsize = 16
max_gen_len = 50

method = 'top_k'
top_k = 30
top_p = 0.5
num_beams = 5
temperature = 0.5
num_return = 6


# GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", return_dict = True).to(device)
# GPT2 parameter setting
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token # to avoid an error

# load trigger data file
df_trigger = pd.read_csv('data/trigger_data.csv', index_col = 0)

# generate result storage
succeed = {'trigger': [], 'word': [], 'generate':[]}
trigger_list = []
word_list = []
generated_list = []


for i in tqdm(range(261*batchsize, len(df_trigger), batchsize)):
# for i in range(144, len(df_trigger), batchsize):
    # print(i)
    # generation setting
    batch = df_trigger.loc[i: i+batchsize-1, :]
    batch_word = list(batch['word'])
    batch_list = list(batch['trigger'])
    batch_lens = list(batch['length'])
    try:
        # generate sentences
        outputs = GPT2_batch_generation(batch_list, max_gen_len, top_p, num_beams, method, top_k, temperature, num_return)


        # extract sentences for each word
        for j in range(len(batch)):
            trigger = batch_list[j]
            word = batch_word[j]
            num = batch_lens[j]
            for k in range(num_return*j, num_return*j+num_return):
                s = extract_sent(str(outputs[k]), word, num)
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
            df_succeed.to_csv('data/succeed_batch.csv')
    except:
        print(i)