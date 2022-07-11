# python data_augmentation.py --trigger_name trigger_data.csv --generate_name augment_result_06251547.csv
# python metrics.py --refer_name slang_augment_30000_split.csv --eval_name augment_result_06251547.csv --metric freqency
# python metrics.py --refer_name slang_augment_30000_split.csv --eval_name augment_result_06251547.csv --metric perplexity
# python metrics.py --refer_name slang_augment_30000_split.csv --eval_name augment_result_06251547.csv
# python train.py --n_epochs_mlm 10 --n_epochs_cls 10

# baseline cls with only linear layer
python train.py --baseline True --mlm_adapter_name model_1657482929 --n_epochs_cls 100 --lr_cls 1e-5 --baseline_with_adapter False --message "baseline cls with only linear layer"

# baseline cls with adapter layer
python train.py --baseline True --mlm_adapter_name model_1657482929 --n_epochs_cls 100 --lr_cls 1e-5 --baseline_with_adapter True --message "baseline cls with adapter layer"

# mlm + cls with adapter not updated
python train.py --baseline False --mlm_adapter_name model_1657482929 --n_epochs_cls 100 --lr_cls 1e-5 --update_adapter_cls False --message "mlm + cls with adapter not updated"

# mlm + cls with adapter updated
python train.py --baseline False --mlm_adapter_name model_1657482929 --n_epochs_cls 100 --lr_cls 1e-5 --update_adapter_cls True --message "mlm + cls with adapter updated"
