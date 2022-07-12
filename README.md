# Environment Build Up
For Linux system, use the command below to build up the conda environment.

```
conda env create -f environment.yml
```

# Slang-Detection
Slang Detection, semantic course, ethz

# Data extraction
TODO:
1. optimize the distribution of the words

# Data Augmentation

method: GPT2 + top_k/top_p/beam

need to test which method and which parameter is better. 

generation speed: about 4min on CPU, batchsize = 32.

TODO:
1. Find a method to test the generation quality(BLEU, diversity:word freqency, perplexity, hugging face, human evaluation, lexical differences, MAUVE). 
2. Do grid search / random search on the parameters, find the dataset with best quality. write script
3. random 100 augmented data for human evaluation

(OPTIONAL)
1. If we can pretrain on GPT2 and then generate.
2. Generate on other datasets.


# MLM and CLS

TODO:
1. add evaluation and test process.
2. test on GPU.
3. add LR scheduler and different opt method.
4. add some special mask for slang.



lr
weight decay
batch size 4-8 比较好
validation+early stop

1. 主结果

2. test数据按sample个数分类

3. baseline + blank adapter  --> 证明slang确实放进了adapter里面

4. tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertlarge, bert --- 对不同大小的模型都能起作用

5. adapter size的影响(config: para: dimension reduct)




step1: 划分数据 ty
step2：MLM: freeze bert全部参数，跑一个bert base的，跑bert large，跑bert mini---于是得到三个adapter
step3：load adapter跑bert base，跑bert large，跑bert mini 逐个调参
step4:load empty adapter跑baseline
step5:不load adapter 跑另一个baseline
ty调参写报告


1. 