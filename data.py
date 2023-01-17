from constant import CHECK_POINT
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


class Dataset:
    def __init__(self, name='imdb'):
        self.name = name
        self.dataset = load_dataset(self.name)

    def build_dataset(self, test_size=0.2):
        dataset = concatenate_datasets([self.dataset['train'], self.dataset['test']])
        dataset = dataset.shuffle(seed=42)

        split_dataset = dataset.train_test_split(test_size=test_size, stratify_by_column="label")
        train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

        auto_tokenizer = AutoTokenizer.from_pretrained(CHECK_POINT)
        tokenize = lambda row: auto_tokenizer(row['text'], truncation=True, padding='max_length', max_length=512)

        tokenized_train_dataset, tokenized_test_dataset = train_dataset.map(tokenize, batched=True), test_dataset.map(tokenize, batched=True)

        tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return tokenized_train_dataset, tokenized_test_dataset
