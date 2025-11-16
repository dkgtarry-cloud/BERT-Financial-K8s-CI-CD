## 1.项目简介

本项目构建了一个基于 BERT + PyTorch 的金融文本情感分类模型，并通过 Flask API 提供推理服务。模型及服务进一步被封装为 Docker 镜像，最终部署到 Kubernetes 集群，并支持：

自动扩缩容（HPA）

CI/CD 自动部署（Jenkins + Harbor）

可观测性监控（Prometheus + Grafana）

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

设置优化器 optimizer 与学习率调度器 scheduler

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

<img width="865" height="145" alt="image" src="https://github.com/user-attachments/assets/13a5ac9b-648f-4c20-9ba9-1e7074c7b01d" />
<br>
结果符合预期






