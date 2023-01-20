import os
from argparse import ArgumentParser
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from constant import CHECK_POINT
from data import Dataset
from utils import compute_metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--output-dir", default='./model/output', type=str)
    parser.add_argument("--save-model-dir", default='./model/test_model', type=str)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    # print('Github: ${accout}')
    # print('Email: ${email}')
    # print('---------------------------------------------------------------------')
    # print('Training ${name} model with hyper-params:') # FIXME
    # print('===========================')
    dataset = Dataset()
    tokenized_train_dataset, tokenized_test_dataset = dataset.build_dataset(test_size=args.test_size)
    
    distlbert_cls = DistilBertForSequenceClassification.from_pretrained(CHECK_POINT, num_labels=2)
    trainer = Trainer(model=distlbert_cls, train_dataset=tokenized_train_dataset)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        per_device_eval_batch_size=8,
        evaluation_strategy='steps',
        eval_steps=300,
        logging_steps=300,
        gradient_accumulation_steps=1,
    )

    trainer = Trainer(
        model=distlbert_cls, 
        args=training_args, 
        train_dataset=tokenized_train_dataset, 
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics
    )

    print("---------------------Training-------------------")
    trainer.train()
    print("---------------------Evaluate-------------------")
    trainer.evaluate()
    trainer.save_model(args.save_model_dir)
    print('---done---')

    
    