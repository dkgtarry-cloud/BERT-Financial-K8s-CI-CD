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
逐行解析原始 TXT 文件

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

### 4.3 模型推理




