from utils import read_json, save_txt


def is_wrong(s1, s2_list, limit=1):
    def is_override(p, g):
        p_list = p.split(' ')
        g_list = g.split(' ')
        j = 0
        for i in range(len(p_list)-1, -1, -1):
            if p_list[i:] == g_list[:j+1] and j+1 >= limit:
                return True
        return False
    s1 = s1.lower()
    for s2 in s2_list:
        s2 = s2.lower()
        if s1 == s2:
            return False
        if s1 in s2 or s2 in s1:
            return False
        if is_override(s1, s2) or is_override(s2, s1):
            return False
    return True


if __name__ == '__main__':
    gold_data = read_json(f'../../../raw_data/squad1/train.json')
    pred_data = read_json('predict.json')  # qa model outputs on train file
    id2all = {}
    for article in gold_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                q = qa['question']
                q_id = qa['id']
                id2all[q_id] = {
                    'context': context,
                    'question': q,
                    'answers': list(map(lambda x:x['text'], qa['answers']))
                }
    src_target = []
    src_ids = []
    if isinstance(list(pred_data.values())[0], list):
        # TODO 需要先进行一次去冗余
        for q_id, pred_list in pred_data.items():
            for pred in pred_list:
                golds = id2all[q_id]['answers']
                if is_wrong(pred, golds, limit=1):
                    src_target.append(id2all[q_id]['context'].replace('\n','') + ' [SEP] ' + pred)
        save_txt(src_target, '../../filter_files/bert-1/train.n_select_wrong.txt')
    else:
        for q_id, pred in pred_data.items():
            golds = id2all[q_id]['answers']
            if is_wrong(pred, golds, limit=1):
                src_target.append(id2all[q_id]['context'].replace('\n', '') + ' [SEP] ' + pred)
                src_ids.append(q_id)
        save_txt(src_target, '../../filter_files/attack/train.select_wrong.txt')
        save_txt(src_ids, '../../filter_files/attack/train.select_wrong.ids.txt')

    print(f'train: {len(id2all)} -> {len(src_target)}')

