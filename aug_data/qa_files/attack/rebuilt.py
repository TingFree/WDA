import os
import random
import argparse
from utils import read_txt, save_json, read_json


# 准确率是多少？
def get_startPos(answer, context):
    return context.find(answer)

# 将两个squad格式的数据合并打乱
def mix_train(t1, t2, st):
    dt1 = read_json(t1)
    dt2 = read_json(t2)
    dt1['data'].extend(dt2['data'])
    random.shuffle(dt1['data'])
    save_json(dt1, st)

# 用于将select_wrong和argu构造成squad格式
def format_squad_1():
    data_type = ['train']
    pref = 1  # 用来标识扩充的数据属于哪一批次
    postfix = 1
    for dt in data_type:
        target = {'data': [], 'version': 'v4.0'}
        src_data = read_txt(os.path.join('../../filter_files/attack', f'{dt}.select_wrong.txt'))
        tgt_data = read_txt(os.path.join('../../qg_files/attack', 'unilm.preds.txt'))
        skip = 0  # 误差数据
        for i in range(len(src_data)):
            try:
                context = src_data[i].split('[SEP]')[0].strip()
                answer = src_data[i].split('[SEP]')[1].strip()
            except:
                skip += 1
                continue
            question = tgt_data[i]
            if not question:
                continue
            target['data'].append({
                'title': '',
                'paragraphs': [{
                    'context': context,
                    'qas': [{
                        'question': question,
                        'answers': [{
                            'answer_start': get_startPos(answer, context),
                            'text': answer
                        }],
                        'id': str(pref) + '_' + str(postfix)
                    }]
                }]
            })
            postfix += 1
        print(f"rebuilt {dt}: {len(target['data'])}, other skip: {skip}")
        save_json(target, os.path.join('..', f'{dt}.before_quality.json'))

# 用于将textMatch过滤之后的数据重构成squad格式
def format_squad_2():
    data_type = ['train']
    data_root = 'textMatch'
    for dt in data_type:
        target = {'data':[], 'version':'v5.0'}
        raw_data = read_json(f'{dt}.before_quality.json')
        id_data = read_txt(os.path.join(data_root, f'{dt}.before_quality.ids.txt'))
        pred_data = read_txt(os.path.join(data_root, f'squad1.format.train.preds.txt'))
        for i, pred in enumerate(pred_data):
            if pred == '1':
                raw_id = raw_data['data'][i]['paragraphs'][0]['qas'][0]['id']
                if raw_id != id_data[i]:
                    print('misMatch...')
                target['data'].append(raw_data['data'][i])
        print(f"{len(pred_data)} -> {len(target['data'])}")
        save_json(target, os.path.join(data_root, f'squad1.{dt}.after_quality.json'))
        mix_train('../../../raw_data/squad1/train.json', f'{data_root}/squad1.{dt}.after_quality.json',
                  f'{data_root}/squad1.{dt}.mix.after_quality.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--switch', default=0)
    args = parser.parse_args()
    if args.switch == 0:
        format_squad_1()  # 用于将select_wrong和aug构造成squad格式, train.before_quality.json
    else:
        format_squad_2()  # 用于将textMatch过滤之后的数据重构成squad格式, train.after_quality.json, train.mix.after_quality.json

