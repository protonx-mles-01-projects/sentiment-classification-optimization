import os
<<<<<<< HEAD
from argparse import ArgumentParser
=======
import sentencepiece
from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from data import Dataset
from utils import compute_metrics

>>>>>>> 738ff3d (Fix model)

if __name__ == "__main__":
    parser = ArgumentParser()
    
<<<<<<< HEAD
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
=======
    # Arguments users used when running command lines
    parser.add_argument("--check-point", default='distilbert-base-uncased', type=str,
                        help='Choose a pre-trained model for fine-tuning.')
    parser.add_argument("--train-dataset", default='./dataset/train.csv.', type=str,
                        help='Directory of train dataset. Only CSV file can be accepted.')
    parser.add_argument("--validation-dataset", default='./dataset/validation.csv', type=str,
                        help='Directory of validation dataset. Only CSV file can be accepted') 
    parser.add_argument("--test-dataset", default='./dataset/test.csv', type=str,
                        help='Directory of test dataset. Only CSV file can be accepted') 
    parser.add_argument("--test-size", default=0.2, type=float,
                        help='test_size is only used in the case you want concatenat all train/test dataset and then split with your ratio.')                  
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--learning-rate", default=2e-5, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--output-dir", default='./model/output', type=str)
    parser.add_argument("--save-model-dir", default='./model/test_model', type=str)
>>>>>>> 738ff3d (Fix model)

    home_dir = os.getcwd()
    args = parser.parse_args()

<<<<<<< HEAD
    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')
    
    # FIXME
    # Do Prediction


=======
    # Project Description
    print('---------------------Welcome to ${name}-------------------')
    # print('Github: ${accout}')
    # print('Email: ${email}')
    # print('---------------------------------------------------------------------')
    # print('Training ${name} model with hyper-params:') # FIXME
    # print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')
    
    # Load model & tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.check_point, num_labels=2, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(args.check_point)

    # Load dataset from path
    ds = Dataset()
    if args.test_size:
        dataset = ds.build_dataset(args.train_dataset, args.validation_dataset, args.test_dataset)
        train_dataset, test_dataset = ds.split_dataset(dataset, args.test_size)
        train_dataset, validation_dataset = ds.split_dataset(train_dataset, args.test_size)
    else:
        train_dataset = ds.build_dataset(args.train_dataset)
        validation_dataset = ds.build_dataset(args.validation_dataset)
        test_dataset = ds.build_dataset(args.test_dataset)

    # Tokenizer inputs
    def tokenization(example):
        return tokenizer(example["text"], padding=True, truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenization, batched=True)
    tokenized_validation_dataset = validation_dataset.map(tokenization, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenization, batched=True)

    # Set hyperparameters
    #logging_steps = len(tokenized_train_dataset) // args.batch_size
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay = args.weight_decay,
        evaluation_strategy='steps',
        #logging_steps=logging_steps,
        log_level="error",
        disable_tqdm=False,
        load_best_model_at_end=True,
        save_total_limit=2
        )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        test_dataset=tokenized_test_dataset, 
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    print("---------------------Training-------------------")
    trainer.train()
    print("---------------------Evaluate-------------------")
    trainer.evaluate()
    trainer.save_model(args.save_model_dir)
    print('---done---')

    
    
>>>>>>> 738ff3d (Fix model)
