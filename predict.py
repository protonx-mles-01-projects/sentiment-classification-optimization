import os
from argparse import ArgumentParser
import torch
from constant import CHECK_POINT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", default='./model/DistilBert', type=str)
    args = parser.parse_args()

    print('---------------------Welcome to ProtonX DistilBert-------------------')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    loaded_model.to(device)

    device_index = loaded_model.device.index if loaded_model.device.type != 'cpu' else -1
    auto_tokenizer = AutoTokenizer.from_pretrained(CHECK_POINT)

    classifier = TextClassificationPipeline(
        model=loaded_model,
        tokenizer=auto_tokenizer, 
        device=device_index
    )
    loaded_model.config.id2label = {0: 'Negative', 1: 'Positive'}

    text = input("Input text to analyze sentiment: ")
    while text != 'q':
        sent, prob = classifier(text)[0]['label'], 100*classifier(text)[0]['score']
        print(f"{sent} sentiment: {prob:.2f}%")
        text = input("Input text to analyze sentiment: ")
    

