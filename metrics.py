from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import list_metrics, load_metric
import pandas as pd
import torch

# load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", return_dict = True).to(device)

# metrics to choose
metrics_list = list_metrics()
print(metrics_list)

# load data
filedir = "data/"
data_cleaned = pd.read_csv(filedir+'data_cleaned_new.csv', index_col=0).sort_values(['word'])
data_generated = pd.read_csv(filedir+'succeed_batch.csv', index_col = 0)

# delete repeated sentences
data_cleaned = data_cleaned.drop_duplicates(keep='first')
data_generated = data_generated.drop_duplicates(keep='first')

# arrange data into the form to use the metrics
dict_refer = data_cleaned.groupby('word')['example'].apply(list).to_dict()
dict_eval = data_generated.groupby('word')['generate'].apply(list).to_dict()
predictions = []
references = []
for key in list(dict_eval.keys()):
    preds = dict_eval[key]
    refers = dict_refer[key]

    tok_refers = [tokenizer.tokenize(refer) for refer in refers]
    references += [tok_refers]*len(preds)
    tok_preds = [tokenizer.tokenize(pred) for pred in preds]
    predictions += tok_preds

# use BLEU metric
metric = load_metric('bleu')
final_score = metric.compute(predictions=predictions, references=references)

final_score