from transformers import AutoTokenizer
from datasets import load_metric
# from evaluate import load
import pandas as pd
import torch
from configuration import CONSTANTS as C
from configuration import Configuration
from collections import defaultdict


class COMPUTE_BLEU:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.data_refer = pd.read_csv(C.DATA_DIR+self.config.refer_name, index_col=0).sort_values(['word']).drop_duplicates(keep='first').reset_index()
        self.data_eval = pd.read_csv(C.DATA_DIR+self.config.eval_name, index_col = 0).drop_duplicates(keep='first').reset_index()
        # data_cleaned = pd.read_csv(C.DATA_DIR+'data_cleaned_new.csv', index_col=0).sort_values(['word'])
        # data_generated = pd.read_csv(C.DATA_DIR+'succeed_batch.csv', index_col = 0)
        self.dict_refer = self.data_refer.groupby('word')['example'].apply(list).to_dict()
        self.dict_eval = self.data_eval.groupby('word')['generate'].apply(list).to_dict()
        self.predictions = []
        self.references = []
        self.score = None

    def set_input(self):
        # arrange data into the form to use the metrics

        for key in list(self.dict_eval.keys()):
            try:
                preds = self.dict_eval[key]
                refers = self.dict_refer[key]

                tok_refers = [self.tokenizer.tokenize(refer) for refer in refers]
                self.references += [tok_refers]*len(preds)

                tok_preds = [self.tokenizer.tokenize(pred) for pred in preds]
                self.predictions += tok_preds
            except:
                continue

    def compute_metric(self):
        self.set_input()
        print('metric input set')
        metric = load_metric('bleu')
        self.score = metric.compute(predictions=self.predictions, references=self.references)
        return self.score


class COMPUTE_PERPLEXITY:
    def __init__(self, config):
        self.config = config
        self.data_eval = pd.read_csv(C.DATA_DIR+self.config.eval_name, index_col = 0).drop_duplicates(keep='first').reset_index()
        self.input_texts = []
        self.score = None

    def set_input(self):
        # arrange data into the form to use the metrics
        dict_eval = self.data_eval.groupby('word')['generate'].apply(list).to_dict()

        for key in list(dict_eval.keys()):
            preds = dict_eval[key]
            self.input_texts += preds

    def compute_metric(self):
        self.set_input()
        print('metric input set')
        # results = perplexity.compute(predictions=predictions, model_id='gpt2')
        metric = load_metric('perplexity', module_type="metric")
        self.score = metric.compute(model_id = 'gpt2', input_texts = self.input_texts, batch_size = 16, device = 'cpu')
        return self.score



class COMPUTE_FREQENCY:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.data_eval = pd.read_csv(C.DATA_DIR+self.config.eval_name, index_col = 0).drop_duplicates(keep='first').reset_index()
        self.list_sents = list(self.data_eval['generate'])
        self.input_texts = []
        self.score = None
        self.length = len(self.data_eval)
        self.word_freqs = defaultdict(int)

    def set_input(self):
        # arrange data into the form to use the metrics
        list_tokens = [self.tokenizer.tokenize(sent) for sent in self.list_sents]

        for text in list_tokens:
            for word in text:
                self.word_freqs[word] += 1

    def compute_metric(self):
        self.set_input()
        return len(self.word_freqs), len(self.word_freqs)/self.length, self.word_freqs


def get_metric(config):
    # use BLEU metric
    if config.metric == 'bleu':
        bleu = COMPUTE_BLEU(config)
        score = bleu.compute_metric()
        # print(score)
        return score
    if config.metric == 'perplexity':
        perplexity = COMPUTE_PERPLEXITY(config)
        score = perplexity.compute_metric()

        # print(score)
        return score
    if config.metric == 'frequency':
        count_freq = COMPUTE_FREQENCY(config)
        count, count_rate, freq = count_freq.compute_metric()
        # print(count)
        # print(freq)
        return count, count_rate, freq


if __name__ == '__main__':
    config = Configuration.parse_cmd()
    get_metric(config)