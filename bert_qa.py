import os
import json
import torch
from simpletransformers.question_answering import QuestionAnsweringModel
from evaluate import in_eval

def create_parentDir(path, exist_ok=True):
    head, tail = os.path.split(path)
    os.makedirs(head, exist_ok=exist_ok)

def read_data(train_file, dev_file, test_file=None):
    train_data = json.load(open(train_file, encoding='utf-8'))
    train_data = [item for topic in train_data['data'] for item in topic['paragraphs']]
    dev_data = json.load(open(dev_file, encoding='utf-8'))
    dev_data = [item for topic in dev_data['data'] for item in topic['paragraphs'] ]
    if test_file:
        test_data = json.load(open(test_file, encoding='utf-8'))
        test_data = [item for topic in test_data['data'] for item in topic['paragraphs']]
        return train_data, dev_data, test_data
    else:
        return train_data, dev_data

def save_json(data, file):
    create_parentDir(file)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print(f'data -> {file}')

def split_preds(preds):
    submission = {}
    n_submission = {}
    a_str, a_pro = preds
    for i in range(len(a_str)):
        assert a_str[i]['id'] == a_pro[i]['id']
        id = a_str[i]['id']
        a = sorted(zip(a_str[i]['answer'], a_pro[i]['probability']), key=lambda x: x[1], reverse=True)[0][0]
        submission[id] = a
        n_submission[id] = a_str[i]['answer']
    return submission, n_submission

train_args = {
    'n_gpu': 2,
    'learning_rate': 5e-5,
    'max_seq_length': 384,
    'max_answer_length': 30,
    'doc_stride': 128,
    'num_train_epochs': 2,
    'train_batch_size': 24,
    'eval_batch_size': 24,
    'gradient_accumulation_steps': 1,
    'warmup_ratio': 0.0,
    'manual_seed': 42,
    'do_lower_case': True,
    'reprocess_input_data': True,
    'output_dir': 'outputs/',
    'save_model_every_epoch': False,
    'save_eval_checkpoints': False,
    'save_optimizer_and_scheduler': True,
    'save_steps': -1, # -1 is disable
    'overwrite_output_dir': True,
    'evaluate_during_training': False,
    'best_model_dir': 'outputs3/best/'
}

bert_base_uncased_file = '../../pretrained_data/bert-base-uncased'

os.environ['CUDA_VISIBLE_DEVICES']="6,7"
train_args['n_gpu'] = torch.cuda.device_count()

## search train and eval without num ********************************
lrs = [3e-5]  # 3e-5, 5e-5, 7e-5
num_epoch = 2
gradient_accumulation_step = 1
batch_sizes = [12] # 6, 12, 24
train_file, dev_file, test_file = 'data/train.json', 'data/dev.json', 'data/test.json'
train_data, dev_data, test_data = read_data(train_file, dev_file, test_file)

for batch_size in batch_sizes:
    for lr in lrs:
        if lr == 3e-5 and batch_size == 24:
            continue
        if lr == 7e-5 and batch_size == 6:
            continue
        output_path = f'outputs' + str(lr) + '_' + str(batch_size*gradient_accumulation_step)
        # if os.path.exists(output_path):
        #     continue
        train_args['output_dir'] = output_path
        train_args['learning_rate']  = lr
        train_args['num_train_epochs'] = num_epoch
        train_args['gradient_accumulation_steps'] = gradient_accumulation_step
        train_args['train_batch_size'] = batch_size
        model = QuestionAnsweringModel('bert', bert_base_uncased_file, args=train_args)
        model.train_model(train_data, eval_data=None)
        model.eval_model(dev_data, output_dir=f'{output_path}/eval/')
        preds, n_preds = split_preds(model.predict(test_data))
        os.makedirs(f'{output_path}/pred', exist_ok=True)
        save_json(preds, f'{output_path}/pred/predict.json')
        # save_json(n_preds, f'{output_path}/pred/n_predict.json')
        print(f"lr: {lr}, batch:{batch_size*gradient_accumulation_step}, last eval: {in_eval(dev_file, f'{output_path}/eval/predictions_test.json')}")
        print(f"lr: {lr}, batch:{batch_size*gradient_accumulation_step}, last test: {in_eval(test_file, f'{output_path}/pred/predict.json')}")
## end *************************************
