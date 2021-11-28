# Wrong-based Data Augmentation

## introduction

the source code for NLPCC2021 《面向问答领域的数据增强方法》/《Data Augmentation Method for Question Answering》

## environment

> pip install -r requirements.txt  # python>=3.6

maybe not complete

> raw_data and unilm is big dir (200MB) to upload, you can try link：https://pan.baidu.com/s/1XL1jOlv4v5UfYHVnVZnMWg password: nfr9

## how to run

1.  train QA model

   ```python
   # train bert qa and get predic, set parameters inside
   python bert_qa.py
   ```

   when you train qa model over, set test_file to train_file, get model outputs on train_file to data augmentation later.

2. get wrong predict and generate questions

   ```python
   # cd aug_data/qa_files/xxx, here xxx is 'attack'
   # get wrong predict file with aug_data/filter_files/xxx/train.select_wrong.txt
   python filter.py
   # cd unilm/src
   # download pretrained model from https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin with bert-large-cased config to unilm/pretrained_data/unilm1_large_cased
   # download checkpoint from https://drive.google.com/open?id=1JN2wnkSRotwUnJ_Z-AbWwoPdP53Gcfsn to unilm/recent_model/microsoft_release
   # generate questions based on selected wrong prediction, saved in unilm/recent_model/microsoft_release/qg_model.bin.train
   bash job.sh
   # manually move qg_model.bin.train to aug_data/qg_files/xxx/unilm.preds.txt
   # cd aug_data/qa_files/attack
   # convert data format for QAMatch
   python rebuilt.py --switch 0
   ```

   

3. filter with QAMatch/roundTrip

   ```python
   # first, you shold train QAMatch Model
   # follow raw_data/README.md to get data
   # cd QAMatch
   # QAMatch pretrained model: https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad, you should set this in QAMatch/train.py before train model
   python train.py  # use squad1 default, you can change it inside
   # cd aug_data/qa_files/attack
   # convert data format for QAMatch
   python for_textMatch.py
   
   # cd QAMatch
   # use QAMatch model to judge
   python test.py
   
   # cd aug_data/qa_files/attack
   python rebuilt.py --switch 1
   # finally, you will get textMatch/train.after_quality.json, mix up train.json to train.mix.after_quality.json, retrain QA Model use it.
   ```

   