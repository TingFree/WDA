from utils import read_json, save_json
from rebuilt import mix_train

model_preds = read_json('roundTrip/predict.json')
aug_data = read_json('train.before_quality.json')

target = {'data': [], 'version': '1.1'}
for article in aug_data['data']:
    answer = article['paragraphs'][0]['qas'][0]['answers'][0]['text']
    q_id = article['paragraphs'][0]['qas'][0]['id']
    if answer == model_preds[q_id]:
        target['data'].append(article)

save_json(target, 'roundTrip/rt_after_quality.json')
print(f"{len(aug_data['data'])} -> {len(target['data'])}")
mix_train('../../../raw_data/squad1/train.json', 'roundTrip/rt_after_quality.json', 'roundTrip/train.mix.rt.json')