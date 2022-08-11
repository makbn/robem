import os

import torch
import torch.nn as nn
from tokenizers import AddedToken
from torch.nn.functional import cosine_similarity
from transformers import BertModel, RobertaModel
from transformers import BertTokenizer, RobertaTokenizer



class DeepSet(nn.Module):
    def __init__(self, dim_input, dim_output=2, dim_hidden=128, pool='max'):
        super(DeepSet, self).__init__()
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_output))

        self.pool = pool

    def forward(self, X):
        X = self.enc(X)
        if self.pool == "max":
            X = X.max(dim=1)[0]
        elif self.pool == "mean":
            X = X.mean(dim=1)
        elif self.pool == "sum":
            X = X.sum(dim=1)
        X = self.dec(X)
        return X

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_size):
        super().__init__()
        self.dense = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(input_size, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class SimpleClassifier(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleClassifier, self).__init__()
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, enc):
        return self.fc(enc)


class DeepClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(DeepClassifier, self).__init__()
        self.drop1 = nn.Dropout(dropout)
        self.hw = Highway(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.th1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.drop3 = nn.Dropout(dropout)
        self.th2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.drop4 = nn.Dropout(dropout)
        self.th3 = nn.Tanh()
        self.classifier = nn.Linear(hidden_size // 4, 2)

    def forward(self, x):
        out = self.drop1(x)
        out = self.hw(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.fc2(self.th1(out))
        out = self.drop3(out)
        out = self.fc3(self.th2(out))
        out = self.drop4(out)
        out = self.classifier(self.th3(out))
        return out


class Highway(nn.Module):

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        proj_result = nn.functional.relu(self.proj(input))
        proj_gate = nn.functional.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


class ResNetBlock(nn.Sequential):

    def __init__(self, input_size, num_layers=2, dropout=0.2, *args):
        super().__init__(*args)
        self.input_size = input_size
        for n in range(num_layers):
            self.add_module("fc-{}".format(n), nn.Linear(input_size, input_size))
            self.add_module("drp-{}".format(n), nn.Dropout(dropout))

    def forward(self, input):
        seq_out = super().forward(input)
        return torch.add(seq_out, input)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.best_f1 = -1
        self.best_loss = 99999

    def reset_weights(self):
        for ch in self.children():
            if hasattr(ch, 'reset_parameters'):
                ch.reset_parameters()

    def get_lm(self):
        return {
            'bert': 'bert-base-uncased',
            'bert-large': 'bert-large-uncased',
            'roberta-base': 'roberta-base',

        }[self.lm_name]

    def get_lm_dim(self):
        return {
            "bert": 768,
            "bert-large": 1024,
            "roberta-base": 768,
        }[self.lm_name]

    def get_lm_class(self):
        return {
            'bert': BertModel,
            'bert-large': BertModel,
            'roberta-base': RobertaModel,
        }[self.lm_name]

    def has_type_token(self):
        return {
            'bert': True,
            'bert-large': True,
            'roberta-base': False,
        }[self.lm_name]

    @staticmethod
    def _get_b4t_answer_num(goal):
        return {"ufet": 10331, "onto": 89, "figer": 113, "bbn": 56}[goal]

    @staticmethod
    def get_tokenizers(lm, add_special_token=True):
        if lm == 'bert' or lm == 'bert-large':
            name = 'bert-base' if lm == 'bert' else 'bert-large'
            tokenizer = BertTokenizer.from_pretrained('{}-uncased'.format(name))
        else:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        if add_special_token:
            print('special token added to tokenizers')
            tokenizer.add_special_tokens({'additional_special_tokens': [AddedToken('ATTR')]})

        return tokenizer

    def save_to_file(self, args, overwrite=True, dataset_name=None):

        if dataset_name:
            dsn = dataset_name
        else:
            dsn = args.dataset_name
        os.makedirs(args.save_dir, exist_ok=True)
        name = "{}-{}-{}-{}-{}-model.pt".format(args.lm, dsn, 'deep' if args.deep else 'simple', args.loss
                                                , 'da' if args.da else 'no')
        full_path = os.path.join(args.save_dir, name)

        if not os.path.exists(full_path) or overwrite:
            torch.save(self, full_path)
            print('model checkpoint saved on: {}'.format(full_path))
        else:
            print('model checkpoint failed to save on: {}! a file with the same name exists!'.format(full_path))

        return full_path


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss





