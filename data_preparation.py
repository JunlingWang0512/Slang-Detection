import pandas as pd
import random
from configuration import CONSTANTS as C

def data_cls_csv():
    train_sl = pd.read_csv("data/slang_train_10000_split.csv")
    train_st = pd.read_csv("data/standard_train_10000.csv")
    test_sl = pd.read_csv("data/slang_test_10000_split.csv")
    test_st = pd.read_csv("data/standard_test_10000.csv")

    train_sl["label"] = 1
    train_st["label"] = 0
    test_sl["label"] = 1
    test_st["label"] = 0

    train_sl = train_sl[['example', 'label']]
    train_st = train_st[['train', 'label']]
    test_sl = test_sl[['example', 'label']]
    test_st = test_st[['test', 'label']]

    train_st.columns = ['example', 'label']
    test_st.columns = ['example', 'label']

    eval_sl = test_sl[:5000]
    eval_st = test_st[:5000]
    test_sl = test_sl[5000:]
    test_st = test_st[5000:]

    trainset = pd.concat([train_sl,train_st], axis = 0).reset_index(drop = True)
    evalset = pd.concat([eval_sl,eval_st], axis = 0).reset_index(drop = True)
    testset = pd.concat([test_sl,test_st], axis = 0).reset_index(drop = True)

    trainset.to_csv('data/train_cls.csv')
    evalset.to_csv('data/eval_cls.csv')
    testset.to_csv('data/test_cls.csv')


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
    filedir = C.DATA_DIR + "slang_augment_30000_split.csv"
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

def augment_split_csv():
    filedir = C.DATA_DIR + 'augment_result_06251547.csv'
    data_augment = pd.read_csv(filedir, index_col=0)  
    random.seed(122)
    sample_idx = random.sample(range(0, data_augment.shape[0]), k=data_augment.shape[0])
    train_cnt = int(data_augment.shape[0]* 0.8)
    train = data_augment.iloc[sample_idx[:train_cnt]]
    eval = data_augment.iloc[sample_idx[train_cnt:]]
    train.to_csv(C.DATA_DIR + 'augment_train.csv')
    eval.to_csv(C.DATA_DIR + 'augment_eval.csv')


if __name__ == '__main__':
    data_cls_csv()
    data_trigger_csv()
    augment_split_csv()
