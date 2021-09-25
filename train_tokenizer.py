import os
import json
from tokenizers import BertWordPieceTokenizer

dataset_path = 'train_sentences.txt'
model_name = 'dziribert'

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

TOKENS_TO_ADD = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '<S>', '<T>']

# Customize training
tokenizer.train(files=[dataset_path], 
                vocab_size=50_000, 
                min_frequency=2, 
                special_tokens=TOKENS_TO_ADD)

tokenizer.enable_truncation(max_length=512)

# save tokenizer vocabulary
os.mkdir(model_name)
tokenizer.save_model(model_name)

# Save tokenizer config
fw = open(os.path.join(model_name, 'tokenizer_config.json'), 'w')
json.dump({"do_lower_case": True, 
            "unk_token": "[UNK]", 
            "sep_token": "[SEP]", 
            "pad_token": "[PAD]", 
            "cls_token": "[CLS]", 
            "mask_token": "[MASK]", 
            "model_max_length": 512, 
            "max_len": 512}, fw)
fw.close()