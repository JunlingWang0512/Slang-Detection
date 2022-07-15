from importlib.metadata import distribution
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import requests

def randomWord():
    response = requests.get("https://api.urbandictionary.com/v0/random")

    return response.text

def specificWord(word):
    response = requests.get("https://api.urbandictionary.com/v0/define?term=" + word)

    return response.text

def getData(num):
    word_count = 0
    dataset = list()

    while word_count < num:
        randomData = json.loads(randomWord())['list']
        for randomEntry in randomData:
            word = randomEntry['word']
            if bool(re.match("^[a-zA-Z\s]*$", word)):
                word_count += 1
                data = json.loads(specificWord(word))['list']
                for entry in data:
                    if entry['word'].lower() in entry['example'].lower():
                        dataset.append({'word': entry['word'],
                                    'defid': entry['defid'], 
                                    'definition': entry['definition'], 
                                    'written_on': entry['written_on'],
                                    'example': entry['example'],
                                    'thumbs_up': entry['thumbs_up'], 
                                    'thumbs_down': entry['thumbs_down']
                                })


    df = pd.DataFrame(dataset)
    df.drop_duplicates(inplace=True)

    # Calculate upvote/downvote ratio
    df['thumbs_down'] = df['thumbs_down'].abs()
    df['ratio'] = df.thumbs_up/df.thumbs_down
    df.loc[np.isinf(df['ratio']), 'ratio'] = df['thumbs_up']
    df['ratio'].fillna(0, inplace=True)

    return df

def sent_split(sent):
    sent1 = sent.split('\n')
    sent2 = []
    for x in sent1:
        sent2 = sent2 + x.split('\r')
    sent3 = [x for x in sent2 if x != '']
    return sent3

def list_sent(row):
    sents = [sent for sent in row['example'] if row['word'] in sent]
    return sents

def extract_example(data):
    data_example = pd.DataFrame(columns = ['word', 'example'])
    count = 0
    for index, row in data.iterrows():
        for example in row['example']:
            data_example.loc[count] = [row['word'], example]
            count += 1

def preprocessData(df):
    filtered = pd.merge(df[df.ratio >= 2], df[df.thumbs_up >= 20])
    filtered = filtered.loc[:, ['word', 'example']]
    filtered['word'] = filtered['word'].str.lower()
    filtered['example'] = filtered['example'].str.lower()
    filtered['example'] = filtered['example'].str.replace('[\[\]]', '')
    filtered['example'] = filtered['example'].str.replace('[“”]', '"')
    filtered['example'] = filtered['example'].str.replace('[‘’]', '\'')
    filtered['example'] = filtered['example'].str.encode('ascii','ignore').str.decode('ascii')
    filtered['example'] = filtered['example'].apply(lambda x: sent_split(x))
    filtered['example'] = filtered.apply(lambda x: list_sent(x), axis = 1)
    filtered_example = extract_example(filtered)
    filtered_example['word'] = filtered_example['word'].fillna('nan')
    filtered_example.drop_duplicates(inplace=True)

    return filtered_example

def addWordIndex(word):
    return word_dict[word]

def countExamples(df):
    count = df['word'].value_counts().rename_axis('word').reset_index(name='count')
    merge = df.merge(count, how='inner', on='word')

    return merge

# Examine the distribution of upvote, downvote and their ratio
distribution_data = getData(50000)

ratio = distribution_data['ratio'].tolist()
plt.hist(ratio, bins=3000, histtype='step')
plt.yscale('log')
plt.xlabel('upvote/downvote ratio')
plt.savefig('up_downvote ratio.png', dpi=300)

up = distribution_data['thumbs_up'].tolist()
down = distribution_data['thumbs_down'].tolist()
plt.hist(up, label='upvote', histtype='step')
plt.hist(down, label='downvote', bins=3000, histtype='step')
plt.legend(loc='upper right')
plt.yscale('log')
plt.xlabel('vote')
plt.savefig('up_downvote.png', dpi=300)

# Obtain data for augmentation
augmentation_data = getData(250000)
augmentation_example = preprocessData(augmentation_data)
word = augmentation_example['word'].tolist()
word_dict = dict.fromkeys(word)

distinct_word_count = 0
for key in word_dict.keys():
    word_dict[key] =  distinct_word_count
    distinct_word_count += 1

augmentation_example['word_idx'] = augmentation_example['word'].apply(addWordIndex)
augmentation_example = augmentation_example.reset_index(drop=True)
multi_example = augmentation_example.groupby('word_idx').filter(lambda x: len(x) > 1)

remain_count = 50000 - len(multi_example.groupby('word_idx').count())
remain_idex = set(range(len(word_dict))).difference(set(multi_example['word_idx'].tolist()))
random.seed(216)
sample_remain = random.sample(list(remain_idex), k=remain_count)
remain_example = augmentation_example[augmentation_example['word_idx'].isin(sample_remain)]
augmentation = pd.concat([multi_example, remain_example], ignore_index=True)
augmentation.drop(columns=['word_idx'], inplace=True)
augmentation.to_csv('slang_augment_50000_updated.csv', index=False)

# Obtain data for classification
classification_data = getData(150000)
classification_example = preprocessData(classification_data)
classification = pd.concat([augmentation, classification_example]).drop_duplicates(keep=False)
classification = classification.reset_index(drop=True)
random.seed(122)
sample_classification = random.sample(range(len(classification)), k=20000)

train = classification.iloc[sample_classification[:10000]]
test = classification.iloc[sample_classification[10000:]]
train_slang = countExamples(train)
test_slang = countExamples(test)
train_slang.to_csv('slang_train_10000.csv', index=False)
test_slang.to_csv('slang_test_10000.csv', index=False)

standard_data = pd.read_csv('word-meaning-examples.csv')
standard_data.drop(columns=['Word', 'Meaning'], inplace=True)
standard_example = sum(standard_data.values.tolist(), [])
standard_example = [example.lower() for example in standard_example if str(example) != 'nan']
random.seed(216)
sample_standard = random.sample(standard_example, k=20000)

train_standard = pd.DataFrame (standard_example[:10000], columns = ['train'])
test_standard = pd.DataFrame (standard_example[10000:], columns = ['test'])
train_standard.to_csv('standard_train_10000.csv', index=False)
test_standard.to_csv('standard_test_10000.csv', index=False)





