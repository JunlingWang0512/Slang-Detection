# Slang-Detection
Slang Detection, semantic course, ethz

# Data Augmentation

method: GPT2 + top_k/top_p/beam

need to test which method and which parameter is better. 

generation speed: about 4min on CPU, batchsize = 32.

TODO:
1. Find a method to test the generation quality(BLEU, diversity:word freqency, hugging face, human evaluation, lexical differences, MAUVE). Do grid search on the parameters, find the dataset with best quality.
2. Delete repeated generation results.

(OPTIONAL)
1. If we can pretrain on GPT2 and then generate.
2. Generate on other datasets.


# MLM and CLS

TODO:
1. add evaluation set and evaluation and test process.
2. test on GPU.
3. add LR scheduler and different opt method.