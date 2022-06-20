# Slang-Detection
Slang Detection, semantic course, ethz

# Data Augmentation

method: GPT2 + top_k/top_p/beam

need to test which method and which parameter is better. 

generation speed: about 4min on CPU, batchsize = 32.

TODO:
1. Find a method to test the generation quality. Do grid search on the parameters, find the dataset with best quality.
2. Delete repeated generation results.
3. If we can pretrain on GPT2 and then generate.
4. Generate on other datasets.