from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification
import joblib

app = Flask(__name__)

# ===== 模型与标签加载 =====
MODEL_NAME = "bert-base-uncased"
model_path = "model/model.pth"
encoder_path = "model/label_encoder.pkl"

# 加载 tokenizer 和标签编码器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_encoder = joblib.load(encoder_path)

# 构建与训练时一致的三分类模型结构
config = BertConfig.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ===== 推理接口 =====
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    label_name = label_encoder.inverse_transform([pred])[0]

    return jsonify({
        "text": text,
        "prediction": label_name
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
