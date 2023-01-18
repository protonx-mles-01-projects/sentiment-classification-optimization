import os
<<<<<<< HEAD
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)

    # FIXME
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')

    # FIXME
    # Do Training
=======
import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


if __name__ == "__main__":
    parser = ArgumentParser()

    # Arguments users used when running command lines
    parser.add_argument("--model-path", default='./model/tests/DistilBert', type=str)
    parser.add_argument("--review", default='I like this film', type=str,
                        help='Add a film review.')

    home_dir = os.getcwd()
    args = parser.parse_args()

    # Project Description
    print('---------------------Welcome to ProtonX DistilBert-------------------')
    

    # Load model & tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    loaded_model.to(device)

    device_index = loaded_model.device.index if loaded_model.device.type != 'cpu' else -1
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    classifier = TextClassificationPipeline(
        model=loaded_model,
        tokenizer=tokenizer, 
        device=device_index
    )

    # Configure labels
    loaded_model.config.id2label = {0: 'Negative', 1: 'Positive'}

    # Predict
    classifier(args.review)

>>>>>>> 738ff3d (Fix model)

