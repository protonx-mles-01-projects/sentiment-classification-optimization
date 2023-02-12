import numpy as np
import torch
from transformers import AutoTokenizer
from constant import CHECK_POINT
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
import onnx
import onnxruntime as ort
import os


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


def softmax(tensor):
    e = np.exp(tensor)
    return e / e.sum()


def export_onnx_file(model, path='model/torch-model.onnx'):
    # set the model to inference mode 
    model.eval() 

    sentence = 'This is a perfect series for family viewing'
    auto_tokenizer = AutoTokenizer.from_pretrained(CHECK_POINT)
    dummy_model_input = auto_tokenizer(sentence, return_tensors="pt")

    torch.onnx.export(
        model,                                          # model being run 
        tuple(dummy_model_input.values()),              # model input
        f=path,                                         # where to save the model  
        # export_params=True,                             # store the trained parameter weights inside the model file 
        input_names=['input_ids', 'attention_mask'],    # the model's input names 
        output_names=['logits'],                        # the model's output names 
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},        # variable length axes 
            'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
            'logits': {0: 'batch_size', 1: 'sequence'}}, 
        do_constant_folding=True,                       # whether to execute constant folding for optimization 
        opset_version=10,                               # the ONNX version to export the model to 
    )



def accuracy_onnx_fn(path, dataset, batch_size=16):
    input_ids = dataset['input_ids'].numpy()
    attention_mask = dataset['attention_mask'].numpy()
    labels = dataset['label'].numpy()

    # ort_sess = ort.InferenceSession(path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(
        path, sess_options=options, providers=["CPUExecutionProvider"]
    )
     
    # disable session.run() fallback mechanism, it prevents for a reset of the execution provider
    # session.disable_fallback() 

    accuracy = 0.0
    total = len(labels)
    n = 0

    while n < total:
        print(n)
        outputs = ort_sess.run(None, {
            'input_ids': input_ids[n:(n+batch_size)],
            'attention_mask': attention_mask[n:(n+batch_size)]
        })

        accuracy += (outputs[0].argmax(-1) == labels[n:(n+batch_size)]).sum()
        n += batch_size

    accuracy = (100 * accuracy / total)
    return accuracy


def quantize_onnx_model(onnx_model_path, quantized_model_path='model/quantize-model.onnx'):    
    # onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)


    print('ONNX full precision model size (MB):', os.path.getsize(onnx_model_path)/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize(quantized_model_path)/(1024*1024))

