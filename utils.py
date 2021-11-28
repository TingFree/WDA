import json
import os
import pandas as pd

def create_parentDir(path, exist_ok=True):
    head, tail = os.path.split(path)
    if not head:
        return
    if not os.path.exists(head):
        print(f'create dir : {head}')
    os.makedirs(head, exist_ok=exist_ok)

def read_json(file):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)
    print(f'{file} -> data')
    return data


def save_txt(data, file):
    create_parentDir(file)
    with open(file, 'w', encoding='utf-8') as f:
        for ex in data:
            f.write(ex + '\n')
    print(f'data -> {file}')


def save_csv(name_data, content_data, file, sep=','):
    create_parentDir(file)
    df = pd.DataFrame(columns=name_data, data=content_data)
    df.to_csv(file, sep=sep)
    print(f'data -> {file}')


def read_txt(file):
    with open(file, encoding='utf-8') as f:
        data = [line.strip() for line in f]
    print(f'{file} -> data')
    return data


def save_json(data, file):
    create_parentDir(file)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print(f'data -> {file}')


def count_squad(file_or_data):
    if isinstance(file_or_data, str):
        file_or_data = read_json(file_or_data)
    total = 0
    for article in file_or_data['data']:
        for paragraph in article['paragraphs']:
            total += len(paragraph['qas'])
    print(f'total qas: {total}')
    return total
