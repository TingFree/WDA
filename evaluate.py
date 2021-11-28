""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function

import argparse
import os
import json
import re
import string
import sys
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    finally_total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                finally_total += 1
                if qa["id"] not in predictions:
                    # message = "Unanswered question " + qa["id"] + " will receive score 0."
                    # print(message, file=sys.stderr)
                    finally_total -= 1
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    print(f'finally eval data: {total} -> {finally_total}')
    exact_match = 100.0 * exact_match / finally_total
    f1 = 100.0 * f1 / finally_total

    return {"exact_match": exact_match, "f1": f1}

def in_eval(gold_file, pred_file_or_data):
    dataset = json.load(open(gold_file, encoding='utf-8'))['data']
    if isinstance(pred_file_or_data, dict):
        predictions = pred_file_or_data
    elif os.path.isfile(pred_file_or_data):
        predictions = json.load(open(pred_file_or_data, encoding='utf-8'))
    else:
        raise ValueError('need correct pred file or pred data')
    return json.dumps(evaluate(dataset, predictions))

def data_eval(gold, pred):
    exact_match = metric_max_over_ground_truths(exact_match_score, pred, gold)
    f1 = metric_max_over_ground_truths(f1_score, pred, gold)
    return exact_match, f1

