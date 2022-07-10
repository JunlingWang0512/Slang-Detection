
import argparse
import json
import os
import pprint
import torch
import time

class Constants(object):
    """
    This is a singleton.
    """
    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.DTYPE = torch.float32
            self.TIME = int(time.time())
            self.DATA_DIR = 'data/'
            self.MODEL_DIR = 'models/model_'+str(self.TIME) + '/'
            self.AUG_ORIGIN_CSV = 'slang_augment_50000_updated.csv'
            self.AUG_TRIGGER_CSV = 'trigger_data.csv'
            self.AUG_RESULT_CSV = 'final_augment.csv'
            self.TRAIN_CLS_CSV = 'cls_train.csv'
            self.EVAL_CLS_CSV = 'cls_eval.csv'
            self.TEST_CLS_CSV = 'cls_test.csv'
            self.TRAIN_MLM_CSV = 'mlm_train.csv'
            self.EVAL_MLM_CSV = 'mlm_eval.csv'
            

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONSTANTS = Constants()


class Configuration(object):

    """Configuration parameters exposed via the commandline."""

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser()

        # General.
        parser.add_argument('--pipeline', default=None, help='Different steps for the project')
        parser.add_argument('--seed', type=int, default=None, help='Random number generator seed.')

        # data augmentation parameters
        parser.add_argument('--generate_name', default = None, help = 'generated data file name')
        parser.add_argument('--model', default = 'GPT2', help = 'augmentation model')
        parser.add_argument('--bs', type = int, default=16, help = 'batch size')
        parser.add_argument('--max_gen_len', type = int, default = 50, help = 'max generation length for augmentation')
        parser.add_argument('--method', default = 'top_k', help = 'top_k, top_p, beam')
        parser.add_argument('--top_k', type = int, default = 30, help = 'top k words')
        parser.add_argument('--top_p', type = float, default = 0.5, help = 'top p')
        parser.add_argument('--num_beams', type = int, default = 5, help = 'beam size')
        parser.add_argument('--temperature', type = float, default = 0.5, help = 'temperature value')
        parser.add_argument('--num_return', type = int, default = 6, help = 'generation number for each trigger')

        # metrics calculation
        parser.add_argument('--metric', default = 'bleu', help = 'bleu, perplexity, freqency.The metric to evaluate the quality of data: bleu, perplexity')
        parser.add_argument('--refer_name', default = None, help = 'reference data file name')
        parser.add_argument('--eval_name', default = None, help = 'data file name to evaluate')

        # adapter enhance
        parser.add_argument('--mlm_threshold', type = float, default = 0.5, help = 'MLM threshold for masking slang words')
        parser.add_argument('--n_epochs_mlm', type = int, default = 50, help = 'mlm task epochs')
        parser.add_argument('--n_epochs_cls', type = int, default = 50, help = 'cls task epochs')
        parser.add_argument('--update_adapter_cls', type = bool, default = True, help = 'whether to update adapter when doing cls')
        parser.add_argument('--enhance', type = bool, default = True, help = 'whether to enhance with adapter when doing cls')

        config = parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)