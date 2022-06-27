import pandas as pd
train1 = pd.read_csv("data/slang_train_10000_split.csv")
train0 = pd.read_csv("data/standard_train_10000.csv")
test1 = pd.read_csv("data/slang_test_10000_split.csv")
test0 = pd.read_csv("data/standard_test_10000.csv")

train1["label"] = 1
train0["label"] = 0
test1["label"] = 1
test0["label"] = 0

train1 = train1[['example', 'label']]
train0 = train1[['train', 'label']]
test1 = test1[['example', 'label']]
test0 = test0[['test', 'label']]

train0.columns = ['example', 'label']
test0.columns = ['example', 'label']

trainset = pd.concat([train1,train0], axis = 0).reset_index(drop = True)
testset = pd.concat([test1,test0], axis = 0).reset_index(drop = True)

trainset.to_csv('data/train_cls.csv')
trainset.to_csv('data/test_cls.csv')


trainset = trainset.drop(columns=['B', 'C'])
