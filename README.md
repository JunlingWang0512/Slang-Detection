# Model Enhancement with Data Augmentation for Slang Detection
Computational Semantics Project, ETH Zürich
## Environment Build Up and Activate
For Linux system, use the command below to build up the conda environment and then activate the environment.

```
conda env create -f environment.yml
conda activate csnlp
```

We can use the pipeline.py file to process different parts of the project.

## Data extraction

Use the file data_processing.py

## Data Augmentation

method: GPT2 + top_k/top_p

metrics: bleu, perplexity, frequency

to get the final augmentation examples.

```
python pipeline.py --pipeline final_augmentation
```

## Mask Language Modelling
run rs_mlm.py for random search for parameters for MLM.

run run_mlm.py for final mask language modellig adpater training and storing.

## Classification

### Baseline Model
run rs_cls_baseline.py for random search baseline.

run cls_baseline_base.py and cls_baseline_mini.py for final baseline.

### Enhanced Model

run rs_cls_enhanced.py for random search enhanced model.

run cls_enhanced_base.py and cls_enhanced_mini.py for final enhanced model.