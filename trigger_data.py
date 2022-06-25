import pandas as pd


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


filedir = "data/slang_augment_30000_split.csv"
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
df_trigger.to_csv('data/trigger_data.csv', index = False)