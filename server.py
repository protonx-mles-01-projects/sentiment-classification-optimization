import os
import io
from waitress import serve
import flask
from flask import Flask, Response, make_response, request, jsonify
from flask_cors import CORS
from waitress import serve
import json


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

# TODO 4: Khởi tạo ứng dụng ở cổng 80
if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 80))
    except:
        port = 80
    print("Starting server on port {}".format(port))
    serve(app, host='0.0.0.0', port=port)