import os
import io
from waitress import serve
import flask
from flask import Flask, Response, make_response, request, jsonify
from flask_cors import CORS
import json
import torch
import predict as pred
from constant import CHECK_POINT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# TODO 1
# Khởi tạo app flask

app = Flask(__name__)
CORS(app)

# TODO 2: Tạo route /api/v1/model_info
# Khi client gọi GET. Trả về acc và loss của model
@app.route('/api/v1/model_info', methods=['GET'])
def get_model_info():
    result_dict = {
        "accuracy": 100.0,
        "loss": 0.0
    }
    return jsonify(result_dict)


# TODO 2: Tạo route /api/v1/predict
# Khi client gọi POST, lay noi dung { "review": "xxx" } tu request body va tra ve ket qua prediction sentiment va score 
@app.route('/api/v1/predict', methods=['POST'])
def post_predict_v1():
    # get model path and load model
    model_path = './model/DistilBert'
    classifier = pred.base_model_pred_fn(model_path)

    # get data from request
    data = request.get_json()
    review = data.get('review', '')
    if not review:
        return jsonify({ "error": "review text is not provided" }), 400
    
    result = classifier(review, padding=True, truncation=True)
    sent, prob = result[0]['label'], 100*result[0]['score']
    result_dict = {
        "score": round(prob, 2),
        "sentiment": sent,
    }
    return jsonify(result_dict)

@app.route('/api/v2/predict', methods=['POST'])
def post_predict_v2():
    # get model path and load model
    model_path = './model/quantize-model.onnx'
    classifier = pred.onnx_pred_fn(model_path)

    # get data from request
    data = request.get_json()
    review = data.get('review', '')
    if not review:
        return jsonify({ "error": "review text is not provided" }), 400
    
    result = classifier(review)
    sent, prob = result[0]['label'], 100*result[0]['score']
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