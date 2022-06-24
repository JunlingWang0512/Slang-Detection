import requests
import json
import pandas as pd
import re
import random

def randomWord():
    response = requests.get("https://api.urbandictionary.com/v0/random")

    return response.text

def specificWord(word):
    response = requests.get("https://api.urbandictionary.com/v0/define?term=" + word)

    return response.text

def sent_split(sent):
    sent1 = sent.split('\n')
    sent2 = []
    for x in sent1:
        sent2 = sent2 + x.split('\r')
    sent3 = [x for x in sent2 if x != '']
    return sent3


# extract example from the list
def list_sent(row):
    sents = [sent for sent in row['example'] if row['word'] in sent]
    if len(sents) > 0:
        return sents[0]
    else:
        return ''

# extract words and examples from API
wordCount = 0
wordList = list()
dataset = list()


while wordCount < 100000:
    randomData = json.loads(randomWord())['list']
    for randomEntry in randomData:
        word = randomEntry['word']
        if word not in wordList and bool(re.match("^[a-zA-Z\s]*$", word)):
            if wordCount % 1000 == 0:
                print(wordCount)
            wordList.append(word)
            data = json.loads(specificWord(word))['list']
            for entry in data:
                if entry['word'] in entry['example']:
                    wordCount += 1
                    dataset.append({'word': entry['word'],
                                'defid': entry['defid'], 
                                'definition': entry['definition'], 
                                'written_on': entry['written_on'],
                                'example': entry['example'],
                                'thumbs_up': entry['thumbs_up'], 
                                'thumbs_down': entry['thumbs_down']
                            })

df = pd.DataFrame(dataset)

filtered = pd.merge(df[df.thumbs_up/df.thumbs_down >= 2], df[df.thumbs_up > 20])

filtered = filtered[~filtered['word'].isnull()]
filtered.drop_duplicates(inplace=True)

filtered = filtered[['word', 'example']]


filtered['word'] = filtered['word'].str.lower()
filtered['example'] = filtered['example'].str.lower()
filtered['example'] = filtered['example'].str.replace('[\[\]]', '')
filtered['example'] = filtered['example'].str.replace('[“”]', '"')
filtered['example'] = filtered['example'].str.replace('[‘’]', '\'')
filtered['example'] = filtered['example'].apply(lambda x: sent_split(x))
filtered['example'] = filtered.apply(lambda x: list_sent(x), axis = 1)
filtered['example'] = filtered['example'].str.encode('ascii','ignore').str.decode('ascii')

filtered = filtered[~(data['example'] == '')]
data.reset_index(drop=True)


# train, test, augment split
random.seed(122)
samples = random.sample(range(0, filtered.shape[0]), k=70000)


train = data.iloc[samples[:10000]]
test = data.iloc[samples[10000:20000]]
augment = data.iloc[samples[20000:]]

train.to_csv('slang_train_10000_split.csv', index=False)
test.to_csv('slang_test_10000_split.csv', index=False)
augment.to_csv('slang_augment_50000_split.csv', index=False)