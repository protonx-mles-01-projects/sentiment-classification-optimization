import os
from argparse import ArgumentParser
from xml.dom import ValidationErr
import torch
from constant import CHECK_POINT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import onnx
import onnxruntime as ort
from utils import softmax


def onnx_pred_fn(path):
    onnx_model = onnx.load(path)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s" % e)
    else:
        print("The model is valid!")
    
    # ort_sess = ort.InferenceSession(path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(
        path, sess_options=options, providers=["CPUExecutionProvider"]
    )
    # ort_sess.disable_fallback() 
    
    auto_tokenizer = AutoTokenizer.from_pretrained(CHECK_POINT)

    def pred_fn(input):
        classes = ['Negative', 'Positive']

        tokenized_input = auto_tokenizer(input, return_tensors="np", truncation=True, padding='max_length', max_length=512)

        outputs = ort_sess.run(None, {
            'input_ids': tokenized_input['input_ids'],
            'attention_mask': tokenized_input['attention_mask']
        })

        result = [{
            'label': classes[outputs[0][0].argmax(-1)],
            'score': softmax(outputs[0])[0].max()
        }]
        return result

    return pred_fn


def base_model_pred_fn(path):
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.config.id2label = {0: 'Negative', 1: 'Positive'}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_index = model.device.index if model.device.type != 'cpu' else -1
    model.to(device)

    auto_tokenizer = AutoTokenizer.from_pretrained(CHECK_POINT)
    classifier = TextClassificationPipeline(
        model=model,
        tokenizer=auto_tokenizer, 
        device=device_index
    )

    return classifier


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", default='./model/DistilBert', type=str)
    args = parser.parse_args()

    print('---------------------Welcome to ProtonX -------------------')

    if os.path.splitext(args.model_path)[1] == '.onnx':
        classifier = onnx_pred_fn(args.model_path)
    else:
        classifier = base_model_pred_fn(args.model_path)


    text = input("Input text to analyze sentiment: ")
    while text != 'q':
        result = classifier(text)
        sent, prob = result[0]['label'], 100*result[0]['score']
        print(f"{sent} sentiment: {prob:.2f}%")

        text = input("Input text to analyze sentiment: ")
    

