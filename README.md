## 1.项目简介

本项目构建了一个基于 BERT + PyTorch 的金融文本情感分类模型，并通过 Flask API 提供推理服务。模型及服务进一步被封装为 Docker 镜像，最终部署到 Kubernetes 集群，并支持：

- 自动扩缩容（HPA）

- CI/CD 自动部署（Jenkins + Harbor）

- 可观测性监控（Prometheus + Grafana）

该项目模拟真实企业中的 AI 推理服务全链路工程化流程：
数据 → 模型训练 → 推理 API → Docker 化 → K8s 部署 → 自动伸缩 → CI/CD → 监控


## 2.项目架构

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/dfafd0ee-0200-4f62-ac07-62245c874823" />
<br> 


## 3.文件结构

```bash
FINANCIAL_NLP_PROJECT/
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── Sentences_AllAgree.txt
├── model/
│   ├── model.pth                # BERT 权重
│   ├── label_encoder.pkl        # 标签编码器
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
├── app.py                       # Flask 推理 API
├── train.py                     # 模型训练脚本
├── data_preparation.py          # 数据清洗脚本
├── Download_model.py            # 下载预训练模型（可选）
├── Dockerfile                   # Docker 镜像构建文件
├── Jenkinsfile                  # CI/CD 流水线
└── requirements.txt
```


## 4.模型训练

### 4.1 数据清洗

数据来自 HuggingFace 的 Financial PhraseBank，其中的 Sentences_AllAgree 文件包含了标注一致性最高的金融文本样本。每条数据的格式如下：

```bash
"Construction volumes meanwhile grow at a rate of 10-15 percent annually .@positive"
```
文本内容紧接一个情感标签（@positive / @negative / @neutral）。
<br> 
<img width="865" height="734" alt="image" src="https://github.com/user-attachments/assets/b0e89825-80a3-41d6-8918-617ebdffd18d" />
<br> 
数据准备脚本（data_preparation.py）完成了以下任务：

- 逐行解析原始 TXT 文件

- 移除标签标记（@positive / @negative / @neutral）

- 生成标准化结构：text、label 两列

- 转换为模型可直接使用的DataFrame 结构化数据

- 按 80/20 进行分层切分成80% → 训练集（train.csv）、20% → 验证集（val.csv）

最终生成：

```bash
data/train.csv
data/val.csv
```

<img width="865" height="205" alt="image" src="https://github.com/user-attachments/assets/c20607f7-79c2-469b-84b3-957d3b33aada" />
<br> 




### 4.2 模型训练

项目使用 BERT-base-uncased 作为文本编码模型，并在其基础上微调金融文本情感分类任务（positive / negative / neutral）。训练脚本位于 train.py。

#### 4.2.1 模型数据加载

训练脚本使用 FinancialDataset 类对文本进行编码：

```bash
encodings = tokenizer(text,
                      truncation=True,
                      padding="max_length",
                      max_length=128,
                      return_tensors="pt")
```

该步骤完成：

- tokenization：将文本拆成 BERT 的 token

- 转换 token IDs：映射为数字序列

- 生成 attention_mask：标记有效 token

- 固定长度（padding/truncation）：保证 batch 尺寸一致

随后由 DataLoader 生成 batch 数据用于训练。

#### 4.2.2 模型定义

使用 HuggingFace Transformers 提供的预训练模型：

```bash
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_)
)
```

此处完成：

- 加载预训练 BERT 模型（encoder 部分参数已训练）

- 创建随机初始化的分类头（Linear 层）

- 设置输出类别数量 num_labels=3（即nn.Linear(768, 3)）

设置优化器 optimizer 与学习率调度器 scheduler。

#### 4.2.3 训练循环

训练过程包括前向传播、反向传播和参数优化：

```bash
outputs = model(**inputs)
loss = outputs.loss
loss.backward()

optimizer.step()
scheduler.step()
optimizer.zero_grad()
```
同时每个 epoch 结束后会在验证集上计算准确率：

```bash
preds = torch.argmax(outputs.logits, dim=1)
acc = (preds == labels).float().mean()
```

确保模型在训练过程中不断评估性能，避免过拟合。

训练过程中会输出：

- 每个 batch 的 loss

- 每个 epoch 的平均 loss

- 验证集准确率

<img width="865" height="303" alt="image" src="https://github.com/user-attachments/assets/b518e1b7-da62-4f75-8b12-23d6592abad0" />
<br>

### 4.3 模型输出与测试

训练结束后，模型权重与标签编码器会保存到 model/ 目录

保存代码：

```bash
torch.save(model.state_dict(), "model/model.pth")
joblib.dump(label_encoder, "model/label_encoder.pkl")
```

这些文件将在推理 API (app.py) 中被加载。

模型推理测试：

```bash
predict.py 'Xiaomi car sales reach a new high, with users generally expressing high satisfaction'
```
<br>
<img width="865" height="145" alt="image" src="https://github.com/user-attachments/assets/13a5ac9b-648f-4c20-9ba9-1e7074c7b01d" />

<br>

结果符合预期。
<br>

## 5.模型推理 API

本项目使用 Flask 封装 BERT 文本分类模型，提供 REST 风格推理接口，实现在线预测能力。

推理代码位于 app.py。

### 5.1 模型加载逻辑

推理服务启动时会加载三个核心组件：

1）Tokenizer（文本编码器）

来自与训练一致的预训练模型：

```bash
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

用于将用户输入的文本转成：

- input_ids

- attention_mask

2）标签编码器（label encoder）

用于将预测的数字类别映射回文字标签（positive / negative / neutral）

3）模型结构 + 训练权重

推理时重新构建与训练阶段一致的 BERT 结构：

```bash
config = BertConfig.from_pretrained(MODEL_NAME, num_labels=3)
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
model.eval()
```

流程：

- 先构建一个新的 BERT 分类模型结构

- 再加载训练阶段保存的权重（model.pth）

- model.eval() 关闭 dropout，使推理稳定


### 5.2 推理接口

API 格式：

- 方法：POST

- 路径：/predict

- 输入 JSON：

```bash
{
  "text": "Li Auto Automobile sales have broken through a new high..."
}
```

- 返回 JSON：

```bash
{
  "text": "Li Auto Automobile sales have broken through a new high...",
  "prediction": "positive"
}
```
推理逻辑：

```bash
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()
label_name = label_encoder.inverse_transform([pred])[0]
```

### 5.3 Docker 部署

Dockerfile：

```bash
FROM python:3.14-slim

WORKDIR /app
COPY . .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install flask transformers joblib scikit-learn requests

EXPOSE 5000
CMD ["python", "app.py"]
```

说明：

- 基于官方 Python slim 镜像

- 安装 CPU 版本 PyTorch（轻量且适合 API 推理）

- 安装 Flask、Transformers 等依赖

- 默认启动 Flask API

### 5.4 构建与运行 Docker

1）构建镜像

```bash
docker build -t financial-nlp-api:v1 .
```

<img width="865" height="312" alt="image" src="https://github.com/user-attachments/assets/48c0cc61-4c48-4c4f-a873-5928d18569d4" />
<br>

构建成功

2）运行容器
```bash
docker run -d -p 5000:5000 financial-nlp-api:v1
```

<img width="865" height="35" alt="image" src="https://github.com/user-attachments/assets/d647c1d9-5b3c-4ea9-b433-fdc9c742e467" />

<img width="865" height="24" alt="image" src="https://github.com/user-attachments/assets/0117aba9-4412-469c-8e45-4e806982cc3d" />
<br>

3）测试容器服务

中性新闻示例

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"text":"This stock looks promising"}'
```

返回：

```bash
{"prediction":"neutral","text":"This stock looks promising"}
```

<img width="865" height="20" alt="image" src="https://github.com/user-attachments/assets/96bb4ae3-675a-4a57-81d7-03ab63ebaf7d" />


<img width="865" height="53" alt="image" src="https://github.com/user-attachments/assets/e70649bc-59ee-4e60-87ba-19cd64e55ac1" />



