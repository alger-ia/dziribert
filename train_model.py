from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import BertTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

model_name = 'dziribert'

# Intialize a BERT like model with random weights
config = BertConfig(vocab_size=50000, max_position_embeddings=512)
model = BertForMaskedLM(config=config)

# Load the trained tokenizer
trained_tokenizer = BertTokenizerFast.from_pretrained(model_name)

# prepare datasets
train_dataset = LineByLineTextDataset(
    tokenizer=trained_tokenizer,
    file_path="./train_sentences.txt",
    block_size=64,
)
test_dataset = LineByLineTextDataset(
    tokenizer=trained_tokenizer,
    file_path="./test_sentences.txt",
    block_size=64,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=trained_tokenizer, mlm=True, mlm_probability=0.25
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./algerian_training",
    evaluation_strategy = "steps",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_gpu_train_batch_size=64,
    logging_steps=5000,
    save_steps=5000,
    save_total_limit=2,
)

# Build a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

trainer.save_model(model_name)