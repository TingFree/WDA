# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:30:07 2020

@author: zhaog
"""
import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertModelTest
from utils import test
from data import DataPrecessForSentence

def main(test_file, pretrained_file, gpu_id, saved_file, batch_size=32):
    if gpu_id:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device('cuda')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-wwm-squad', do_lower_case=True)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(bert_tokenizer, test_file, 512)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    print("\t* Building model...")
    model = BertModelTest().to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing BERT model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, auc, preds = test(model, test_loader, device)
    with open(saved_file, 'w', encoding='utf-8') as f:
        for pred in preds:
            f.write(str(pred) + '\n')
        print(f'data -> {saved_file}')
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(batch_time, total_time, (accuracy*100), auc))


if __name__ == "__main__":
    gpu_id = 0
    test_file = '../aug_data/qa_files/attack/textMatch/format.train.tsv'
    saved_file = '../aug_data/qa_files/attack/textMatch/squad1.format.train.preds.txt'
    main(test_file, "models/best.pth.tar", gpu_id, saved_file)