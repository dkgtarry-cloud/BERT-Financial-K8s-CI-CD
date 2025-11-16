import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据
data_path = "data/Sentences_AllAgree.txt"

sentences = []
labels = []

with open(data_path, "r", encoding="latin-1") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if "@positive" in line:
            label = "positive"
            text = line.replace("@positive", "").strip()
        elif "@negative" in line:
            label = "negative"
            text = line.replace("@negative", "").strip()
        elif "@neutral" in line:
            label = "neutral"
            text = line.replace("@neutral", "").strip()
        else:
            continue
        sentences.append(text)
        labels.append(label)

df = pd.DataFrame({"text": sentences, "label": labels})

# 划分训练集 / 验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# 保存
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)

print("✅ 数据准备完成：")
print("训练集样本数：", len(train_df))
print("验证集样本数：", len(val_df))
print("标签分布：")
print(df['label'].value_counts())
