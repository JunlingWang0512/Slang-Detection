# python data_augmentation.py --trigger_name trigger_data.csv --generate_name augment_result_06251547.csv
# python metrics.py --refer_name slang_augment_30000_split.csv --eval_name augment_result_06251547.csv --metric freqency
# python metrics.py --refer_name slang_augment_30000_split.csv --eval_name augment_result_06251547.csv --metric perplexity
# python metrics.py --refer_name slang_augment_30000_split.csv --eval_name augment_result_06251547.csv

python train.py --mlm_train_name augment_train.csv --mlm_eval_name augment_eval.csv --n_epochs_mlm 1 --n_epochs_cls 1