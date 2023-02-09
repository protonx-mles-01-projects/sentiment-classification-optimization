import os
import io
from waitress import serve
import flask
from flask import Flask, Response, make_response, request, jsonify
from flask_cors import CORS
import json
import torch
from constant import CHECK_POINT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# TODO 1
# Khởi tạo app flask

app = Flask(__name__)
CORS(app)

# TODO 2: Tạo route /api/v1/model_info
# Khi client gọi GET. Trả về acc, loss và runtime của model
@app.route('/api/v1/model_info', methods=['GET'])
def get_model_info():
    # open trainer_state, get the infomation of last epoch
    with open('model/trainer_state.json') as file:
        data = json.load(file)
    result_dict = {
        "accuracy": data['log_history'][-1]['eval_accuracy'],
        "loss": data['log_history'][-1]['eval_loss'],
        "run_time":  data['log_history'][-1]['eval_runtime']
    }
    return jsonify(result_dict)


# TODO 2: Tạo route /api/v1/predict
# Khi client gọi POST, lay noi dung { "review": "xxx" } tu request body va tra ve ket qua prediction sentiment va score 
@app.route('/api/v1/predict', methods=['POST'])
def post_predict():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = './model/DistilBert'
    loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    loaded_model.to(device)

    device_index = loaded_model.device.index if loaded_model.device.type != 'cpu' else -1
    auto_tokenizer = AutoTokenizer.from_pretrained(CHECK_POINT)

    classifier = TextClassificationPipeline(
        model=loaded_model,
        tokenizer=auto_tokenizer, 
        device=device_index
    )
    loaded_model.config.id2label = { 0: 'Negative', 1: 'Positive' }

    data = request.get_json()
    review = data.get('review', '')
    if not review:
        return jsonify({ "error": "review text is not provided" }), 400
        
    sent, prob = classifier(review)[0]['label'], 100*classifier(review)[0]['score']
    result_dict = {
        "score": round(prob, 2),
        "sentiment": sent,
    }
    return jsonify(result_dict)

# TODO 4: Khởi tạo ứng dụng ở cổng 80
if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 80))
    except:
        port = 80
    print("Starting server on port {}".format(port))
    serve(app, host='0.0.0.0', port=port)