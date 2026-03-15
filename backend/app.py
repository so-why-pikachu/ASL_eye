import os
import torch
import pymysql
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from src.model import BiLSTMAttentionModel
import src.config as config

app = Flask(__name__)
CORS(app)

# 配置
GLB_ROOT =config.GLB_ROOT
MODEL_PATH =config.MODEL_PATH
DB_CONFIG = {'host': 'localhost', 'user': 'root', 'password': '123456', 'db': 'sign_language_db'}

# 初始化推理引擎
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMAttentionModel(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, num_classes=config.NUM_CLASSES, num_layers=config.NUM_LAYERS,dropout=0)

# 强力加载逻辑
sd = torch.load(MODEL_PATH, map_location=device)
new_sd = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
model.load_state_dict(new_sd, strict=False)
model.to(device).eval()

# --- 2. 接口实现 ---

@app.route('/api/sign/predict', methods=['POST'])
def predict():
    # 1. 接收特征 -> 2. 推理 -> 3. 映射 ID 到 Name
    # 逻辑参考你之前的 pipeline.predict
    return jsonify({"code": 200, "data": {"word_name": "ACCIDENT"}})

@app.route('/api/sign/downloads', methods=['GET'])
def downloads():
    word = request.args.get('name')
    
    # 1. 从 MySQL 查文件夹名
    conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cursor:
        cursor.execute("SELECT folder_name FROM sign_assets WHERE word_name = %s", (word,))
        row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"code": 404, "msg": "Resource not found"}), 404

    folder = row['folder_name']
    abs_path = os.path.join(GLB_ROOT, folder)
    
    # 2. 递归/遍历所有 .glb
    urls = []
    for root, _, files in os.walk(abs_path):
        for f in files:
            if f.endswith('.glb'):
                rel = os.path.relpath(os.path.join(root, f), GLB_ROOT)
                urls.append(f"http://{request.host}/static/models/{rel}")
    
    urls.sort() # 确保帧顺序
    return jsonify({"code": 200, "data": {"urls": urls}})

@app.route('/static/models/<path:filename>')
def serve_glb(filename):
    return send_from_directory(GLB_ROOT, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)