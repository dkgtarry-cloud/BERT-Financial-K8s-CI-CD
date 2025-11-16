import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import os

# ===============================
# 1️⃣ 超参数设置
# ===============================
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 2️⃣ 数据加载类
# ===============================
class FinancialDataset(Dataset):
    def __init__(self, csv_path, tokenizer, label_encoder, max_len=128):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = label_encoder.transform(df["label"].tolist())
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encodings = self.tokenizer(text, truncation=True, padding="max_length",
                                   max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ===============================
# 3️⃣ 读取数据
# ===============================
train_csv = "data/train.csv"
val_csv = "data/val.csv"
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)

label_encoder = LabelEncoder()
label_encoder.fit(df_train["label"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = FinancialDataset(train_csv, tokenizer, label_encoder)
val_dataset = FinancialDataset(val_csv, tokenizer, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===============================
# 4️⃣ 模型定义
# ===============================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer,
                          num_warmup_steps=0, num_training_steps=num_training_steps)

# ===============================
# 5️⃣ 训练循环
# ===============================
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == inputs["labels"]).sum().item()
            total += len(preds)
    acc = correct / total
    return acc

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0

    for batch in loop:
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    val_acc = evaluate()
    print(f"✅ Epoch {epoch+1} 完成 | 平均Loss={total_loss/len(train_loader):.4f} | 验证准确率={val_acc:.4f}")

# ===============================
# 6️⃣ 保存模型与标签映射
# ===============================
torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
import joblib
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
print("✅ 模型与标签映射已保存到 model/ 文件夹。")
