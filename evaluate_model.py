import random
import pandas as pd
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#model = 'bert-base-multilingual-uncased'
#model = 'aubmindlab/bert-base-arabert'
#model = 'qarib/bert-base-qarib'
#model = 'UBC-NLP/MARBERT'
#model = 'CAMeL-Lab/bert-base-arabic-camelbert-da'
#model = 'CAMeL-Lab/bert-base-arabic-camelbert-mix'
model = 'amine/dziribert'

tokenizer = BertTokenizer.from_pretrained(model)

# seeds to reproduce the results of the paper
seeds = [38, 66, 216, 128, 285, 0, 367, 14, 196, 42]

# code to generate new seeds
#seeds = []
#for i in range(10):
#    seeds.append(random.randint(1, 500))

# the results presented on the paper were obtined on Tesla K80 GPU
# please check the type of your GPU using the nvidia-smi command

def get_model():
  return BertForSequenceClassification.from_pretrained(model, num_labels=3)
#  return BertForSequenceClassification.from_pretrained(model, num_labels=10)

def preprocess_function(examples):
  return tokenizer(examples['text'], truncation=True, max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, 
                                                               preds, 
                                                               average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

train_data = pd.read_csv('data/train_sent.csv').drop(['Unnamed: 0'], axis=1)
test_data = pd.read_csv('data/test_sent.csv').drop(['Unnamed: 0'], axis=1)
#train_data = pd.read_csv('data/train_emotion.csv').drop(['Unnamed: 0'], axis=1)
#test_data = pd.read_csv('data/test_emotion.csv').drop(['Unnamed: 0'], axis=1)
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
encoded_train = train_dataset.map(preprocess_function, batched=True)
encoded_test = test_dataset.map(preprocess_function, batched=True)

args = TrainingArguments(
    "sentimentClassification",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

accuracy = 0
f1 = 0
precision = 0
recall = 0
i = 0

for seed in seeds:
    print('seed ', i, ' : ' , seed)
    args.seed = seed
    trainer = Trainer(
        model_init=get_model,
        args=args,
        train_dataset=encoded_train,
        eval_dataset=encoded_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
        )
    trainer.train()
    res = trainer.evaluate()
    accuracy += res['eval_accuracy']
    f1 += res['eval_f1']
    precision += res['eval_precision']
    recall += res['eval_recall']
    i += 1

print(accuracy/len(seeds), 
      f1/len(seeds), 
      precision/len(seeds), 
      recall/len(seeds))