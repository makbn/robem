from time import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from em.models.em_base_model import ASLSingleLabel


class Statistics(object):
    """Accumulator for loss statistics, inspired by ONMT.
    Keeps track of the following metrics:
    * F1
    * Precision
    * Recall
    * Accuracy
    """

    def __init__(self):
        self.loss_sum = 0
        self.examples = 0
        self.tps = 0
        self.tns = 0
        self.fps = 0
        self.fns = 0
        self.start_time = time()

    def update(self, loss=0, tps=0, tns=0, fps=0, fns=0):
        examples = tps + tns + fps + fns
        self.loss_sum += loss * examples
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
        self.examples += examples

    def loss(self):
        return self.loss_sum / self.examples

    def f1(self):
        prec = self.precision()
        recall = self.recall()
        return 2 * prec * recall / max(prec + recall, 1)

    def precision(self):
        return 100 * self.tps / max(self.tps + self.fps, 1)

    def recall(self):
        return 100 * self.tps / max(self.tps + self.fns, 1)

    def accuracy(self):
        return 100 * (self.tps + self.tns) / self.examples

    def examples_per_sec(self):
        return self.examples / (time() - self.start_time)


def calc_f1(pred_labels, step=0.05):
    preds = pred_labels[0]
    labels = pred_labels[1]

    best_th = 0.5
    f1 = 0.0

    for th in np.arange(0.0, 1.0, 0.05):
        pred = [1 if p > th else 0 for p in preds]
        new_f1 = f1_score(labels, pred)
        if new_f1 > f1:
            f1 = new_f1
            best_th = th

    return f1, best_th


def compute_scores(output, target):
    predictions = output
    correct = (predictions == target).float()
    incorrect = (1 - correct).float()
    positives = (target.data == 1).float()
    negatives = (target.data == 0).float()

    tp = torch.dot(correct, positives)
    tn = torch.dot(correct, negatives)
    fp = torch.dot(incorrect, negatives)
    fn = torch.dot(incorrect, positives)
    return tp, tn, fp, fn


def get_criterion(args, device):
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.loss == 'wce':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([args.neg_weight, args.pos_weight]), reduction='mean') \
            .to(device)

    elif args.loss == 'asl':
        criterion = ASLSingleLabel(gamma_neg=args.neg_weight, gamma_pos=args.pos_weight).to(device)
    else:
        raise ValueError('loss is undefined!')

    return criterion
