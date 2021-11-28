from utils import read_json, save_csv, save_txt
import pandas as pd


def count_len():
    data_type = ['train', 'dev', 'test']
    for dt in data_type:
        df = pd.read_csv(f'pq/{dt}.tsv', sep='\t')
        pa = df['text_a'].values
        q = df['text_b'].values
        labels = df['label'].values
        p, n = 0, 0
        for label in labels:
            if label == 1:
                p += 1
            else:
                n += 1
        print(f'num {dt} data: {len(pa)}, positive: {p}, negative:{n}')

if __name__ == '__main__':
    data_type = ['train']
    for dt in data_type:
        raw_data = read_json(f'{dt}.before_quality.json')
        target_data = []
        ids = []
        t = 1
        for article in raw_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    q_a = qa['question'] +' [SEP] '+ qa['answers'][0]['text']
                    if t <= 5:
                        target_data.append([context, q_a, 0])
                        t += 1
                    else:
                        target_data.append([context, q_a, 1])
                    ids.append(qa['id'])
        save_csv(name_data=['text_a', 'text_b', 'label'], content_data=target_data[:len(target_data)], file=f'textMatch/format.{dt}.tsv', sep='\t')
        save_txt(ids, f'textMatch/{dt}.before_quality.ids.txt')
